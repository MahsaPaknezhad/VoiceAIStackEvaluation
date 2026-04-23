# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Literal
from pydantic import BaseModel
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import (
    RTVIObserver,
    RTVIBotLLMTextMessage,
    RTVITextMessageData,
)
from pipecat.frames.frames import LLMTextFrame, FunctionCallInProgressFrame
from loguru import logger


RTVI_PROTOCOL_VERSION = "1.0.0"
RTVI_MESSAGE_LABEL = "rtvi-ai"
RTVIMessageLiteral = Literal["rtvi-ai"]



class RTVILLMFunctionCallInProgressMessageData(BaseModel):
    function_name: str
    tool_call_id: str
    args: dict


class RTVILLMFunctionCallInProgressMessage(BaseModel):
    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["llm-function-call"] = "llm-function-call"
    data: RTVILLMFunctionCallInProgressMessageData


class CustomRTVIObserver(RTVIObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer = ""
        # Current mode can be: None or "thinking"
        self._mode = None

        # Tag definitions
        self._start_tags = {
            "thinking": "<thinking>",
        }
        self._end_tags = {
            "thinking": "</thinking>",
        }
        logger.info("CustomRTVIObserver initialized")

    async def _handle_llm_text_frame(self, frame: LLMTextFrame):
        """Stream-safe stripping of XML-like tags and forwarding only inner text."""
        # logger.info(f"Received LLM text frame: '{frame.text}' (final: {getattr(frame, 'final', False)})")
        
        # Accumulate incoming text
        self._buffer += frame.text
        # logger.debug(f"Buffer now contains: '{self._buffer[:100]}{'...' if len(self._buffer) > 100 else ''}'")

        # Process buffer and emit cleaned chunks
        # If this is the final frame, force emission of any remaining buffer
        is_final = getattr(frame, 'final', False)
        chunks = self._extract_display_chunks(force_emit=is_final)
        # logger.info(f"Extracted {len(chunks)} chunks (final: {is_final})")
        
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            # logger.info(f"Emitting chunk {i}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")
            message = RTVIBotLLMTextMessage(data=RTVITextMessageData(text=chunk))
            await self.push_transport_message_urgent(message)

    def _extract_display_chunks(self, force_emit=False):
        """Generator yielding clean text chunks to display (no tags).

        Args:
            force_emit: If True, emit all remaining buffer content (used when final=True)

        Rules:
        - For <thinking>...</thinking>, completely skip content (don't emit).
        - For all other text, emit as-is.
        - Handle streaming: keep partial tags/content in buffer until complete or
          safe to emit.
        """
        chunks = []

        while True:
            # If we don't have a mode, look for the next start tag
            if self._mode is None:
                start_info = self._find_next_start_tag(self._buffer)
                if start_info is None:
                    # If force_emit is True (final frame), emit everything remaining
                    if force_emit:
                        if self._buffer:
                            chunks.append(self._buffer)
                            self._buffer = ""
                        break
                    
                    # Check if buffer might contain partial start tag
                    max_tag_len = max(len(tag) for tag in self._start_tags.values())
                    
                    # Always emit single characters that can't be start of any tag
                    if len(self._buffer) == 1 and not any(tag.startswith(self._buffer) for tag in self._start_tags.values()):
                        chunks.append(self._buffer)
                        self._buffer = ""
                        break
                    
                    if len(self._buffer) < max_tag_len:
                        # Buffer is small, check if it could be start of a tag
                        could_be_tag = any(tag.startswith(self._buffer) for tag in self._start_tags.values())
                        if could_be_tag:
                            # Might be partial tag, wait for more
                            break
                        else:
                            # Not a tag, emit everything
                            chunks.append(self._buffer)
                            self._buffer = ""
                            break
                    
                    # Buffer is large enough, check if it ends with partial tag
                    has_partial_tag = False
                    for tag in self._start_tags.values():
                        for i in range(1, len(tag)):
                            if self._buffer.endswith(tag[:i]):
                                has_partial_tag = True
                                break
                        if has_partial_tag:
                            break
                    
                    if has_partial_tag:
                        # Keep only the potential partial tag
                        for tag in self._start_tags.values():
                            for i in range(len(tag) - 1, 0, -1):
                                if self._buffer.endswith(tag[:i]):
                                    safe_text = self._buffer[:-i]
                                    if safe_text:
                                        chunks.append(safe_text)
                                    self._buffer = self._buffer[-i:]
                                    return chunks
                    
                    # No partial tag, emit everything
                    if self._buffer:
                        chunks.append(self._buffer)
                        self._buffer = ""
                    break

                tag_key, start_idx, tag_len = start_info
                # Emit everything before the start tag
                if start_idx > 0:
                    chunks.append(self._buffer[:start_idx])
                # Discard everything up to and including the start tag
                self._buffer = self._buffer[start_idx + tag_len:]
                self._mode = self._normalize_mode(tag_key)
                continue

            # We are inside a mode; find its closing tag(s)
            end_tags = self._end_tags_for_mode(self._mode)
            end_idx, matched_end_tag = self._find_earliest(self._buffer, end_tags)

            if end_idx != -1:
                # Found end tag - skip the inner content entirely for thinking mode
                # Consume inner + the specific matched end tag and reset mode
                self._buffer = self._buffer[end_idx + len(matched_end_tag):]
                self._mode = None
                continue

            # No end tag yet; for thinking mode, don't emit anything, just wait
            if self._mode == "thinking":
                if force_emit:
                    # Final frame - clear thinking buffer and reset mode
                    self._buffer = ""
                    self._mode = None
                    break
                
                # Check if we might have a partial end tag - don't emit anything
                holdback = max(
                    self._longest_endtag_prefix_suffix(self._buffer, t) for t in end_tags
                )
                if holdback == len(self._buffer):
                    # Entire buffer might be partial end tag, wait for more
                    break
                else:
                    # Keep some buffer to detect partial end tags
                    max_end_tag_len = max(len(t) for t in end_tags) - 1
                    if len(self._buffer) > max_end_tag_len:
                        # Discard excess content but keep potential partial end tag
                        self._buffer = self._buffer[-max_end_tag_len:]
                    break
            else:
                # For other modes (if any), we'd emit content here
                break

        return chunks

    def _find_next_start_tag(self, text: str):
        """Return (tag_key, index, tag_len) of earliest start tag, or None."""
        earliest = None
        for key, tag in self._start_tags.items():
            idx = text.find(tag)
            if idx == -1:
                continue
            if earliest is None or idx < earliest[1]:
                earliest = (key, idx, len(tag))
        return earliest

    def _normalize_mode(self, tag_key: str):
        return tag_key

    def _end_tags_for_mode(self, mode: str):
        if mode == "thinking":
            return [self._end_tags["thinking"]]
        return []

    def _find_earliest(self, text: str, needles):
        earliest_idx = -1
        earliest_tag = None
        for t in needles:
            idx = text.find(t)
            if idx == -1:
                continue
            if earliest_idx == -1 or idx < earliest_idx:
                earliest_idx = idx
                earliest_tag = t
        return earliest_idx, earliest_tag


    def _longest_endtag_prefix_suffix(self, buffer_text: str, end_tag: str) -> int:
        """Return length of the longest prefix of end_tag that is a suffix of buffer_text."""
        max_prefix = 0
        # Check from 1 to len(end_tag)-1
        max_check = min(len(buffer_text), len(end_tag) - 1)
        for k in range(max_check, 0, -1):
            if buffer_text.endswith(end_tag[:k]):
                max_prefix = k
                break
        return max_prefix

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        direction = data.direction
        
        if frame.id in self._frames_seen:
            return

        if isinstance(frame, FunctionCallInProgressFrame):
            if direction == FrameDirection.DOWNSTREAM:
                await self._handle_function_call_in_progress(frame)
                self._frames_seen.add(frame.id)
        await super().on_push_frame(data)
    
    async def _handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        """Process function call in progress frames for the RTVI client.
        
        This emits the RTVIEvent.LLMFunctionCall event that is consumed by clients.
        """
        message_data = RTVILLMFunctionCallInProgressMessageData(
            function_name=frame.function_name,
            tool_call_id=frame.tool_call_id,
            args=frame.arguments,
        )
        message = RTVILLMFunctionCallInProgressMessage(data=message_data)
        await self.push_transport_message_urgent(message, exclude_none=False)