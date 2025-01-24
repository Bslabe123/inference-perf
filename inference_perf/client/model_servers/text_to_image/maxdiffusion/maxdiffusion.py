from typing import Any
from client.model_servers.text_to_image.text_to_image import (
    Text_To_Image_Model_Server_Client,
)


class MaxDiffusion_Client(Text_To_Image_Model_Server_Client):

    def build_request(self) -> Any:
        pass
        # TODO: Implement me

    def parse_response(self) -> None:
        pass
        # TODO: Implement me
