static string DetermineSystemArchitecture()
        {
            var arch = RuntimeInformation.ProcessArchitecture;
            switch (arch)
            {
                case Architecture.X86:
                    return "x86";
                case Architecture.X64:
                    return "x64";
                case Architecture.Arm:
                    return "arm";
                case Architecture.Arm64:
                    return "arm64";
                default:
                    throw new NotSupportedException();
            }
        }

if (charValue == '\r' || charValue == '\n')
            {
                if (_buffer.Length > 0)
                {
                    _log.WriteLine(_buffer.ToString());
                    _buffer.Clear();
                }

                _currentLog.Append(charValue);
            }
            else

