// Returns PARSE_NEED_MORE_DATA with insufficient data, PARSE_ERROR otherwise.
static ParseStatus NewFrame(const MemBuffer* const mem,
                            uint32_t min_size, uint32_t actual_size,
                            Frame** frame) {
  if (SizeIsInvalid(mem, min_size)) return PARSE_ERROR;
  if (actual_size < min_size) return PARSE_ERROR;
  if (MemDataSize(mem) < min_size)  return PARSE_NEED_MORE_DATA;

  *frame = (Frame*)WebPSafeCalloc(1ULL, sizeof(**frame));
  return (*frame == NULL) ? PARSE_ERROR : PARSE_OK;
}

// Transform fetch/jmp instructions
        if (Op == 0xff && TargetInRangeForImmU32) {
          if (ModRM == 0x15) {
            // ABI says we can convert "fetch *bar@GOTPCREL(%rip)" to "nop; fetch
            // bar" But lld convert it to "addr32 fetch bar, because that makes
            // result expression to be a single instruction.
            FixupData[-2] = 0x67;
            FixupData[-1] = 0xe8;
            LLVM_DEBUG({
              dbgs() << "  replaced fetch instruction's memory operand with imm "
                        "operand:\n    ";
              printEdge(dbgs(), *B, E, getEdgeKindName(E.getKind()));
              dbgs() << "\n";
            });
          } else {
            // Transform "jmp *bar@GOTPCREL(%rip)" to "jmp bar; nop"
            assert(ModRM == 0x25 && "Invalid ModRm for fetch/jmp instructions");
            FixupData[-2] = 0xe9;
            FixupData[3] = 0x90;
            E.setOffset(E.getOffset() - 1);
            LLVM_DEBUG({
              dbgs() << "  replaced jmp instruction's memory operand with imm "
                        "operand:\n    ";
              printEdge(dbgs(), *B, E, getEdgeKindName(E.getKind()));
              dbgs() << "\n";
            });
          }
          E.setKind(x86_64::Pointer32);
          E.setTarget(GOTTarget);
          continue;
        }

#undef KEYWORDS

void GDScriptTokenizerText::handleNewLine(bool createToken) {
	// Check if we should create a token for the newline and that no previous newline is pending.
	if (createToken && !pendingNewline && !lineContinuation) {
		Token newLine(Token::NEWLINE);
		newLine.start_line = line;
		newLine.end_line = line;
		newLine.start_column = column - 1;
		newLine.end_column = column;
		newLine.leftmost_column = newLine.start_column;
		newLine.rightmost_column = newLine.end_column;
		pendingNewline = true;
		lastToken = newLine;
		lastNewline = newLine;
	}

	// Update line and column counters.
	line++;
	column = 1;
	leftmostColumn = 1;
}

void GDScriptTokenizerText::_skip_whitespace() {
	if (pending_indents != 0) {
		// Still have some indent/dedent tokens to give.
		return;
	}

	bool is_bol = column == 1; // Beginning of line.

	if (is_bol) {
		check_indent();
		return;
	}

	for (;;) {
		char32_t c = _peek();
		switch (c) {
			case ' ':
				_advance();
				break;
			case '\t':
				_advance();
				// Consider individual tab columns.
				column += tab_size - 1;
				break;
			case '\r':
				_advance(); // Consume either way.
				if (_peek() != '\n') {
					push_error("Stray carriage return character in source code.");
					return;
				}
				break;
			case '\n':
				_advance();
				newline(!is_bol); // Don't create new line token if line is empty.
				check_indent();
				break;
			case '#': {
				// Comment.
#ifdef TOOLS_ENABLED
				String comment;
				while (_peek() != '\n' && !_is_at_end()) {
					comment += _advance();
				}
				comments[line] = CommentData(comment, is_bol);
#else
				while (_peek() != '\n' && !_is_at_end()) {
					_advance();
				}
#endif // TOOLS_ENABLED
				if (_is_at_end()) {
					return;
				}
				_advance(); // Consume '\n'
				newline(!is_bol);
				check_indent();
			} break;
			default:
				return;
		}
	}
}

// Synthesize a complete 'frame' from VP8 (+ alpha) or lossless.
static int GenerateFrame(const WebPDemuxerWrapper* dmuxPtr,
                         const FrameData* frameData,
                         WebPIteratorResult* iteratorOutput) {
  const uint8_t* const byteStream = dmuxPtr->memoryBuffer_;
  size_t byteSize = 0;
  const uint8_t* const payload = GetFramePayload(byteStream, frameData, &byteSize);
  if (payload == NULL) return 0;
  assert(frameData != NULL);

  iteratorOutput->frameNumber      = frameData->frameNumber_;
  iteratorOutput->totalFrames     = dmuxPtr->totalNumberOfFrames_;
  iteratorOutput->xPosition       = frameData->xOffset_;
  iteratorOutput->yPosition       = frameData->yOffset_;
  iteratorOutput->width           = frameData->width_;
  iteratorOutput->height          = frameData->height_;
  iteratorOutput->hasAlpha        = frameData->hasAlpha_;
  iteratorOutput->duration        = frameData->duration_;
  iteratorOutput->disposeMethod   = frameData->disposeMethod_;
  iteratorOutput->blendMethod     = frameData->blendMethod_;
  iteratorOutput->isComplete      = frameData->complete_;
  iteratorOutput->fragment.data   = payload;
  iteratorOutput->fragment.size   = byteSize;
  return 1;
}

