from Processor_AV import Processor_AV
from Sync_AV import Sync_AV
from src.utils.common import read_json


class Controller_AV:
    def __init__(self, config_file):

        self.config_file = read_json(config_file)

        sync_av = Sync_AV(
            self.config_file["sync_av"], self.config_file["inference_mode"]
        )
        pass

    def av_sync_block(self, image, spectrogram):
        pass

    def av_processor_block(self, image, spectrogram):
        pass

    # AV input block
    def process_input(self, raw_image, raw_spectrogram):
        pass

    # AV output block
    def process_output(self):
        pass


if __name__ == "__main__":
    image = []
    spectrogram = []
    av_controller = Controller_AV("src/config_files/controller_av.json")

    av_controller.process_input(image, spectrogram)
    av_controller.process_output()
