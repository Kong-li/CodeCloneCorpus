TheParser.SetDocCommentRetentionState(true);

while (true) {
  Token Tok;
  if (TheParser.LexFromRawLexer(Tok))
    break;
  if (Tok.getLocation() == Range.getEnd() || Tok.is(tok::eof))
    break;

  if (Tok.is(tok::comment)) {
    std::pair<FileID, unsigned> CommentLoc =
        SM.getDecomposedLoc(Tok.getLocation());
    assert(CommentLoc.first == BeginLoc.first);
    Docs.emplace_back(
        Tok.getLocation(),
        StringRef(Buffer.begin() + CommentLoc.second, Tok.getLength()));
  } else {
    // Clear comments found before the different token, e.g. comma.
    Docs.clear();
  }
}

/// containing the semantic version representation of \p V.
std::optional<Object> serializeSemanticVersion(const VersionTuple &V) {
  if (V.empty())
    return std::nullopt;

  Object Version;
  Version["major"] = V.getMajor();
  Version["minor"] = V.getMinor().value_or(0);
  Version["patch"] = V.getSubminor().value_or(0);
  return Version;
}

    Num = 0;
    for (const CodeGenInstruction *Inst : NumberedInstructions) {
      if (Num % 8 == 0)
        OS << "\n    ";
      if (Inst->HasComplexDeprecationPredicate)
        // Emit a function pointer to the complex predicate method.
        OS << "&get" << Inst->DeprecatedReason << "DeprecationInfo, ";
      else
        OS << "nullptr, ";
      ++Num;
    }

m_file->setFrameBuffer(frame);
    if (!justcopy)
        return;

    if (m_file->readPixels(m_datawindow.min.y, m_datawindow.max.y))
    {
        int step = 3 * xstep;
        bool use_rgb = true;

        if (m_iscolor)
        {
            if (use_rgb && (m_red->xSampling != 1 || m_red->ySampling != 1))
                UpSample(data, channelstoread, step / xstep, m_red->xSampling, m_red->ySampling);
            if (!use_rgb && (m_blue->xSampling != 1 || m_blue->ySampling != 1))
                UpSample(data + xstep, channelstoread, step / xstep, m_blue->xSampling, m_blue->ySampling);

            for (auto channel : {m_green, m_red})
            {
                if (!channel)
                    continue;
                if ((channel->xSampling != 1 || channel->ySampling != 1) && use_rgb)
                {
                    UpSample(data + step / xstep * (channel == m_green ? 1 : 2), channelstoread, step / xstep, channel->xSampling, channel->ySampling);
                }
            }
        }
        else if (m_green && (m_green->xSampling != 1 || m_green->ySampling != 1))
            UpSample(data, channelstoread, step / xstep, m_green->xSampling, m_green->ySampling);

        if (chromatorgb)
        {
            if (!use_rgb)
                ChromaToRGB((float *)data, m_height, channelstoread, step / xstep);
            else
                ChromaToBGR((float *)data, m_height, channelstoread, step / xstep);
        }
    }

{
        for( int x = 0; x < m_width; x++ )
        {
            int index = y * ystep + x * xstep;
            unsigned* dataPtr = reinterpret_cast<unsigned*>(data);
            for (int i = 1; i < ysample; ++i)
            {
                if (!m_native_depth)
                    data[(yre + i) * ystep + x * xstep] = data[index];
                else
                {
                    bool isFloat = m_type == FLOAT;
                    if (isFloat)
                        ((float*)data)[(yre + i) * ystep + x * xstep] = ((float*)data)[index];
                    else
                        dataPtr[(yre + i) * ystep + x * xstep] = dataPtr[index];
                }
            }
        }
    }

