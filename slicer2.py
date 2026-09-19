import os.path
from argparse import ArgumentParser

import numpy as np
import soundfile


# This function is obtained from librosa.
def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(
        y, shape=out_shape, strides=out_strides
    )
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    def _frame_to_sample(self, frame_index: int, total_samples: int) -> int:
        return min(total_samples, frame_index * self.hop_size)

    def _compute_trim_bounds(self, rms_list, total_frames):
        """计算前导/尾随静音修剪后的帧范围。
        前后各最多保留 max_sil_kept 帧静音。
        返回 (trim_start, trim_end)；若全是静音，返回 (0, total_frames)。
        """
        if total_frames == 0:
            return 0, total_frames
        is_silent = rms_list < self.threshold
        non_silent_indices = np.where(~is_silent)[0]
        if len(non_silent_indices) == 0:
            # 全是静音，保持原样返回，避免生成空文件
            return 0, total_frames
        first_non_silent = int(non_silent_indices[0])
        last_non_silent = int(non_silent_indices[-1])
        trim_start = max(0, first_non_silent - self.max_sil_kept)
        trim_end = min(total_frames, last_non_silent + 1 + self.max_sil_kept)
        if trim_start >= trim_end:
            return 0, total_frames
        return trim_start, trim_end

    def slice_ranges(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
            total_samples = waveform.shape[1]
        else:
            samples = waveform
            total_samples = waveform.shape[0]

        rms_list = get_rms(
            y=samples,
            frame_length=self.win_size,
            hop_length=self.hop_size,
        ).squeeze(0)
        return self.slice_ranges_from_rms(rms_list, total_samples)

    def slice_ranges_from_rms(self, rms_list, total_samples):
        if rms_list.shape[0] == 0:
            return [(0, total_samples)]

        total_frames = rms_list.shape[0]
        is_silent = rms_list < self.threshold

        # 1. 计算前导/尾随静音修剪边界
        trim_start, trim_end = self._compute_trim_bounds(rms_list, total_frames)

        # 2. 收集所有连续静音段
        silence_segments = []
        sil_start = None
        for i in range(total_frames):
            if is_silent[i]:
                if sil_start is None:
                    sil_start = i
            else:
                if sil_start is not None:
                    silence_segments.append((sil_start, i - 1))
                    sil_start = None
        if sil_start is not None:
            silence_segments.append((sil_start, total_frames - 1))

        # 3. 若无法切割，返回修剪后的单段
        if not silence_segments or total_frames <= self.min_length:
            return [(self._frame_to_sample(trim_start, total_samples),
                     self._frame_to_sample(trim_end, total_samples))]

        # 4. 计算“往前搜索范围”：5秒对应的帧数（min_length 是目标时长对应的帧数）
        five_sec_frames = self.min_length // 2

        audio_start = trim_start
        audio_end = trim_end
        if audio_start >= audio_end:
            return [(0, total_samples)]

        # 5. 逐个片段寻找切割点
        cut_points = []
        clip_start = audio_start

        while clip_start < audio_end:
            target = clip_start + self.min_length
            if target >= audio_end:
                break

            # 优先往前找：在 [clip_start + 5s, target] 范围内找静音段
            forward_start = clip_start + five_sec_frames
            forward_end = target

            best_cut = None
            best_dist = None
            for sil in silence_segments:
                if sil[1] >= forward_start and sil[0] <= forward_end:
                    overlap_start = max(sil[0], forward_start)
                    overlap_end = min(sil[1], forward_end)
                    sil_mid = (overlap_start + overlap_end) // 2
                    dist = abs(sil_mid - target)
                    if best_dist is None or dist < best_dist:
                        best_cut = sil_mid
                        best_dist = dist

            if best_cut is not None:
                cut_points.append(best_cut)
                clip_start = best_cut
            else:
                # 往前找不到，往后找 target 之后的第一个静音段
                found = False
                for sil in silence_segments:
                    if sil[0] > target:
                        sil_mid = (sil[0] + sil[1]) // 2
                        if sil_mid < audio_end:
                            cut_points.append(sil_mid)
                            clip_start = sil_mid
                            found = True
                        break
                if not found:
                    break

        # 6. 若始终没有切割点，返回修剪后的单段
        if not cut_points:
            return [(self._frame_to_sample(trim_start, total_samples),
                     self._frame_to_sample(trim_end, total_samples))]

        # 7. 根据 cut_points 生成 ranges
        ranges = []
        prev_frame = audio_start
        for cp in cut_points:
            if cp > prev_frame:
                ranges.append((
                    self._frame_to_sample(prev_frame, total_samples),
                    self._frame_to_sample(cp, total_samples),
                ))
                prev_frame = cp
        if prev_frame < audio_end:
            ranges.append((
                self._frame_to_sample(prev_frame, total_samples),
                self._frame_to_sample(audio_end, total_samples),
            ))

        if not ranges:
            return [(0, total_samples)]

        return ranges

    # @timeit
    def slice(self, waveform):
        ranges = self.slice_ranges(waveform)
        if len(waveform.shape) > 1:
            return [waveform[:, begin:end] for begin, end in ranges]
        return [waveform[begin:end] for begin, end in ranges]


def main():
    parser = ArgumentParser()
    parser.add_argument('audio', type=str, help='The audio to be sliced')
    parser.add_argument('--out', type=str, help='Output directory of the sliced audio clips')
    parser.add_argument('--db_thresh', type=float, required=False, default=-40,
                        help='The dB threshold for silence detection')
    parser.add_argument('--min_length', type=int, required=False, default=5000,
                        help='The minimum milliseconds required for each sliced audio clip')
    parser.add_argument('--min_interval', type=int, required=False, default=300,
                        help='The minimum milliseconds for a silence part to be sliced')
    parser.add_argument('--hop_size', type=int, required=False, default=10,
                        help='Frame length in milliseconds')
    parser.add_argument('--max_sil_kept', type=int, required=False, default=500,
                        help='The maximum silence length kept around the sliced clip, presented in milliseconds')
    args = parser.parse_args()
    out = args.out
    if out is None:
        out = os.path.dirname(os.path.abspath(args.audio))
    import librosa
    audio, sr = librosa.load(args.audio, sr=None)
    slicer = Slicer(
        sr=sr,
        threshold=args.db_thresh,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept
    )
    chunks = slicer.slice(audio)
    if not os.path.exists(out):
        os.makedirs(out)
    for i, chunk in enumerate(chunks):
        soundfile.write(os.path.join(out, f'%s_%d.wav' % (os.path.basename(args.audio).rsplit('.', maxsplit=1)[0], i)), chunk, sr)


if __name__ == '__main__':
    main()