from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport


async def create_transport(webrtc_connection=None, runner_args=None):
    transport_params = TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(
                    start_secs=0.5,
                    stop_secs=0.2,
                    # min_volume=0.7,
                )),
                turn_analyzer=LocalSmartTurnAnalyzerV3()
            )
            
    if webrtc_connection:
        return SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=transport_params,
            )

    return await create_transport(runner_args, {'webrtc': lambda: transport_params})
