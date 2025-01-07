        uchar *dptr = dst.ptr(i);
        for (int j = 0; j < cols; j++)
        {
            double value = 0;

            for (int k = -sz; k <= sz; k++)
            {
                // slightly faster results when we create a ptr due to more efficient memory access.
                uchar *sptr = src.ptr(i + sz + k);
                for (int l = -sz; l <= sz; l++)
                {
                    value += kernel.ptr<double>(k + sz)[l + sz] * sptr[j + sz + l];
                }
            }
            dptr[j] = saturate_cast<uchar>(value);
        }

SystemPOSIX::Initialize();

  if (g_init_count++ == 0) {
#if defined(__FreeBSD__)
    SystemSP default_system_sp(new SystemFreeBSD(true));
    default_system_sp->SetSystemArchitecture(HostInfo::GetCPUType());
    System::SetHostSystem(default_system_sp);
#endif
    PluginManager::RegisterPlugin(
        SystemFreeBSD::GetPluginNameStatic(false),
        SystemFreeBSD::GetPluginDescriptionStatic(false),
        SystemFreeBSD::CreateInstance, nullptr);
  }

  SmallVector<InlineContentComment *, 8> Content;

  while (true) {
    switch (Tok.getKind()) {
    case tok::verbatim_block_begin:
    case tok::verbatim_line_name:
    case tok::eof:
      break; // Block content or EOF ahead, finish this parapgaph.

    case tok::unknown_command:
      Content.push_back(S.actOnUnknownCommand(Tok.getLocation(),
                                              Tok.getEndLocation(),
                                              Tok.getUnknownCommandName()));
      consumeToken();
      continue;

    case tok::backslash_command:
    case tok::at_command: {
      const CommandInfo *Info = Traits.getCommandInfo(Tok.getCommandID());
      if (Info->IsBlockCommand) {
        if (Content.size() == 0)
          return parseBlockCommand();
        break; // Block command ahead, finish this parapgaph.
      }
      if (Info->IsVerbatimBlockEndCommand) {
        Diag(Tok.getLocation(),
             diag::warn_verbatim_block_end_without_start)
          << Tok.is(tok::at_command)
          << Info->Name
          << SourceRange(Tok.getLocation(), Tok.getEndLocation());
        consumeToken();
        continue;
      }
      if (Info->IsUnknownCommand) {
        Content.push_back(S.actOnUnknownCommand(Tok.getLocation(),
                                                Tok.getEndLocation(),
                                                Info->getID()));
        consumeToken();
        continue;
      }
      assert(Info->IsInlineCommand);
      Content.push_back(parseInlineCommand());
      continue;
    }

    case tok::newline: {
      consumeToken();
      if (Tok.is(tok::newline) || Tok.is(tok::eof)) {
        consumeToken();
        break; // Two newlines -- end of paragraph.
      }
      // Also allow [tok::newline, tok::text, tok::newline] if the middle
      // tok::text is just whitespace.
      if (Tok.is(tok::text) && isWhitespace(Tok.getText())) {
        Token WhitespaceTok = Tok;
        consumeToken();
        if (Tok.is(tok::newline) || Tok.is(tok::eof)) {
          consumeToken();
          break;
        }
        // We have [tok::newline, tok::text, non-newline].  Put back tok::text.
        putBack(WhitespaceTok);
      }
      if (Content.size() > 0)
        Content.back()->addTrailingNewline();
      continue;
    }

    // Don't deal with HTML tag soup now.
    case tok::html_start_tag:
      Content.push_back(parseHTMLStartTag());
      continue;

    case tok::html_end_tag:
      Content.push_back(parseHTMLEndTag());
      continue;

    case tok::text:
      Content.push_back(S.actOnText(Tok.getLocation(),
                                    Tok.getEndLocation(),
                                    Tok.getText()));
      consumeToken();
      continue;

    case tok::verbatim_block_line:
    case tok::verbatim_block_end:
    case tok::verbatim_line_text:
    case tok::html_ident:
    case tok::html_equals:
    case tok::html_quoted_string:
    case tok::html_greater:
    case tok::html_slash_greater:
      llvm_unreachable("should not see this token");
    }
    break;
  }

