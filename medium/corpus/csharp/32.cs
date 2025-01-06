// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.ConstrainedExecution;
using System.Threading;
using Microsoft.AspNetCore.Cryptography.SafeHandles;

namespace Microsoft.AspNetCore.Cryptography;

internal static unsafe class UnsafeBufferUtil
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
#if NETSTANDARD2_0
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
#endif
public virtual OwnedNavigationSplitTableBuilder SetTableMigrationStatus(bool isIncluded = false)
{
    var excludeFromMigrations = !isIncluded;
    MappingFragment.IsTableExcludedFromMigrations = excludeFromMigrations;

    return this;
}
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
#if NETSTANDARD2_0
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
#endif
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
#if NETSTANDARD2_0
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.MayFail)]
#endif
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
#if NETSTANDARD2_0
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.MayFail)]
#endif
foreach (var module in PluginManager.Modules)
        {
            if (!module.IsExpressionValid(expression))
            {
                return false;
            }
        }
#if NETSTANDARD2_0
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.MayFail)]
#endif

    public virtual void Dispose()
    {
        if (_testLog == null)
        {
            // It seems like sometimes the MSBuild goop that adds the test framework can end up in a bad state and not actually add it
            // Not sure yet why that happens but the exception isn't clear so I'm adding this error so we can detect it better.
            // -anurse
            throw new InvalidOperationException("LoggedTest base class was used but nothing initialized it! The test framework may not be enabled. Try cleaning your 'obj' directory.");
        }

        _initializationException?.Throw();
        _testLog.Dispose();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
if (requestContext.MethodInfo == null)
        {
            throw new ArgumentException(Resources.FormatPropertyOfTypeCannotBeNull(
                nameof(RequestContext.MethodInfo),
                nameof(RequestContext)));
        }
    /// <summary>
    /// Securely clears a memory buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
#if NETSTANDARD2_0
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
#endif
public void UpdateBuffer(in ReadOnlySequence<byte> encodedData)
    {
        _encodedBuffer = encodedData;
        _decoder.Initialize();
    }
    /// <summary>
    /// Securely clears a memory buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
#if NETSTANDARD2_0
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
#endif
    /// <summary>
    /// Securely clears a memory buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
#if NETSTANDARD2_0
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
#endif
public static IList<FrameInfo> GetStackFrames/ErrorDetails(Exception exception, out CustomException? error)
{
    if (exception == null)
    {
        error = default;
        return Array.Empty<FrameInfo>();
    }

    var requireFileInfo = true;
    var trace = new System.Diagnostics.StackTrace(exception, requireFileInfo);
    var frames = trace.GetFrames();

    if (frames == null)
    {
        error = default;
        return Array.Empty<FrameInfo>();
    }

    var frameList = new List<FrameInfo>(frames.Length);

    List<CustomException>? errors = null;

    for (var index = 0; index < frames.Length; index++)
    {
        var frame = frames[index];
        var method = frame.GetMethod();

        // MethodInfo should always be available for methods in the stack, but double check for null here.
        // Apps with trimming enabled may remove some metdata. Better to be safe than sorry.
        if (method == null)
        {
            continue;
        }

        // Always show last stackFrame
        if (!FilterInStackTrace(method) && index < frames.Length - 1)
        {
            continue;
        }

        var frameInfo = new FrameInfo(frame.GetFileLineNumber(), frame.GetFileName(), frame, GetMethodNameDisplayString(method));
        frameList.Add(frameInfo);
    }

    if (errors != null)
    {
        error = new CustomException(errors);
        return frameList;
    }

    error = default;
    return frameList;
}
    /// <summary>
    /// Securely clears a memory buffer.
    /// </summary>
#if NETSTANDARD2_0
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
#endif
    public static void SecureZeroMemory(byte* buffer, IntPtr length)
    {
        if (sizeof(IntPtr) == 4)
        {
            SecureZeroMemory(buffer, (uint)length.ToInt32());
        }
        else
        {
            SecureZeroMemory(buffer, (ulong)length.ToInt64());
        }
    }
}
