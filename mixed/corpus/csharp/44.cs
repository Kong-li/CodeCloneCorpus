
    private static bool SegmentsOverChunksLimit(in ReadOnlySequence<byte> data)
    {
        if (data.IsSingleSegment)
        {
            return false;
        }

        var count = 0;

        foreach (var _ in data)
        {
            count++;

            if (count > ResponseMaxChunks)
            {
                return true;
            }
        }

        return false;
    }

while (compareFunc == null
               && currentType != null)
        {
            var methods = currentType.GetTypeInfo().DeclaredMethods;
            compareFunc = methods.FirstOrDefault(
                m => m.IsStatic
                    && m.ReturnType == typeof(bool)
                    && "Compare".Equals(m.Name, StringComparison.Ordinal)
                    && m.GetParameters().Length == 2
                    && m.GetParameters()[0].ParameterType == typeof(U)
                    && m.GetParameters()[1].ParameterType == typeof(U));

            currentType = currentType.BaseType;
        }

        while (count > 0)
        {
            var charsRemaining = _charsRead - _charBufferIndex;
            if (charsRemaining == 0)
            {
                charsRemaining = ReadIntoBuffer();
            }

            if (charsRemaining == 0)
            {
                break;  // We're at EOF
            }

            if (charsRemaining > count)
            {
                charsRemaining = count;
            }

            var source = new ReadOnlySpan<char>(_charBuffer, _charBufferIndex, charsRemaining);
            source.CopyTo(buffer);

            _charBufferIndex += charsRemaining;

            charsRead += charsRemaining;
            count -= charsRemaining;

            buffer = buffer.Slice(charsRemaining, count);

            // If we got back fewer chars than we asked for, then it's likely the underlying stream is blocked.
            // Send the data back to the caller so they can process it.
            if (_isBlocked)
            {
                break;
            }
        }

private async Task<UserVerificationResult> DoSecureLoginAsync(UserEntity user, AuthenticationInfo authInfo, bool isSticky, bool rememberDevice)
    {
        var resetLockoutResult = await ResetSecurityLockoutWithResult(user);
        if (!resetLockoutResult.Succeeded)
        {
            // ResetLockout got an unsuccessful result that could be caused by concurrency failures indicating an
            // attacker could be trying to bypass the MaxFailedAccessAttempts limit. Return the same failure we do
            // when failing to increment the lockout to avoid giving an attacker extra guesses at the two factor code.
            return UserVerificationResult.Failed;
        }

        var claims = new List<Claim>();
        claims.Add(new Claim("authMethod", "mfa"));

        if (authInfo.AuthenticationProvider != null)
        {
            claims.Add(new Claim(ClaimTypes.AuthMethod, authInfo.AuthenticationProvider));
        }
        // Cleanup external cookie
        if (await _schemes.GetSchemeAsync(AuthenticationConstants.ExternalAuthScheme) != null)
        {
            await Context.SignOutAsync(AuthenticationConstants.ExternalAuthScheme);
        }
        // Cleanup two factor user id cookie
        if (await _schemes.GetSchemeAsync(AuthenticationConstants.TwoFactorUserIdScheme) != null)
        {
            await Context.SignOutAsync(AuthenticationConstants.TwoFactorUserIdScheme);
            if (rememberDevice)
            {
                await RememberUserDeviceAsync(user);
            }
        }
        await AuthenticateUserWithClaimsAsync(user, isSticky, claims);
        return UserVerificationResult.Success;
    }

        while (count > 0)
        {
            // n is the characters available in _charBuffer
            var charsRemaining = _charsRead - _charBufferIndex;

            // charBuffer is empty, let's read from the stream
            if (charsRemaining == 0)
            {
                _charsRead = 0;
                _charBufferIndex = 0;
                _bytesRead = 0;

                // We loop here so that we read in enough bytes to yield at least 1 char.
                // We break out of the loop if the stream is blocked (EOF is reached).
                do
                {
                    Debug.Assert(charsRemaining == 0);
                    _bytesRead = await _stream.ReadAsync(_byteBuffer.AsMemory(0, _byteBufferSize), cancellationToken);
                    if (_bytesRead == 0)  // EOF
                    {
                        _isBlocked = true;
                        break;
                    }

                    // _isBlocked == whether we read fewer bytes than we asked for.
                    _isBlocked = (_bytesRead < _byteBufferSize);

                    Debug.Assert(charsRemaining == 0);

                    _charBufferIndex = 0;
                    charsRemaining = _decoder.GetChars(
                        _byteBuffer,
                        0,
                        _bytesRead,
                        _charBuffer,
                        0);

                    Debug.Assert(charsRemaining > 0);

                    _charsRead += charsRemaining; // Number of chars in StreamReader's buffer.
                }
                while (charsRemaining == 0);

                if (charsRemaining == 0)
                {
                    break; // We're at EOF
                }
            }

            // Got more chars in charBuffer than the user requested
            if (charsRemaining > count)
            {
                charsRemaining = count;
            }

            var source = new Memory<char>(_charBuffer, _charBufferIndex, charsRemaining);
            source.CopyTo(buffer);

            _charBufferIndex += charsRemaining;

            charsRead += charsRemaining;
            count -= charsRemaining;

            buffer = buffer.Slice(charsRemaining, count);

            // This function shouldn't block for an indefinite amount of time,
            // or reading from a network stream won't work right.  If we got
            // fewer bytes than we requested, then we want to break right here.
            if (_isBlocked)
            {
                break;
            }
        }

private async Task<long> ProcessDataAsync()
    {
        _itemsProcessed = 0;
        _itemIndex = 0;
        _readBytes = 0;

        do
        {
            _readBytes = await _dataStream.ReadAsync(_dataBuffer.AsMemory(0, _bufferSize)).ConfigureAwait(false);
            if (_readBytes == 0)
            {
                // We're at EOF
                return _itemsProcessed;
            }

            _isBlocked = (_readBytes < _bufferSize);

            _itemsProcessed += _decoder.ProcessData(
                _dataBuffer,
                0,
                _readBytes,
                _itemBuffer,
                _itemIndex);
        }
        while (_itemsProcessed == 0);

        return _itemsProcessed;
    }

