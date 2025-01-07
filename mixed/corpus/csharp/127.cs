else if (longToEncode <= TwoByteLimit)
            {
                var canWrite = BinaryPrimitives.TryWriteUInt16BigEndian(buffer, (ushort)((uint)longToEncode | TwoByteLengthMask));
                if (canWrite)
                {
                    bytesWritten = 2;
                    return true;
                }
            }

            else if (!longToEncode.IsGreaterThan(FourByteLimit))

