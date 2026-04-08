from faster_whisper import WhisperModel
import time

def transcribe_audio(file_path):
    # 模型规模可选: tiny, base, small, medium, large-v3
    # 建议笔记本先用 small 或 medium 测试
    model_size = "small"

    print(f"正在加载模型: {model_size}...")
    # 如果有 NVIDIA 显卡，把 device 改为 "cuda"，否则用 "cpu"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print("开始识别...")
    start_time = time.time()
    
    # beam_size=5 是官方推荐的平衡参数
    segments, info = model.transcribe(file_path, beam_size=5, language="zh")

    print(f"检测到语言: {info.language} (置信度: {info.language_probability:.2f})")

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

    end_time = time.time()
    print(f"\n识别完成！总耗时: {end_time - start_time:.2f}秒")

if __name__ == "__main__":
    # 替换为你自己录制的音频文件路径
    audio_file = "test_audio.wav" 
    transcribe_audio(audio_file)
