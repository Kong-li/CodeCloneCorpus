// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Linq;
using Microsoft.AspNetCore.Mvc.ApplicationModels;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>
/// Contains the extension methods for <see cref="AspNetCore.Mvc.MvcOptions.Conventions"/>.
/// </summary>
public static class ApplicationModelConventionExtensions
{
    /// <summary>
    /// Removes all application model conventions of the specified type.
    /// </summary>
    /// <param name="list">The list of <see cref="IApplicationModelConvention"/>s.</param>
    /// <typeparam name="TApplicationModelConvention">The type to remove.</typeparam>
    public static void RemoveType<TApplicationModelConvention>(this IList<IApplicationModelConvention> list)
        where TApplicationModelConvention : IApplicationModelConvention
    {
        ArgumentNullException.ThrowIfNull(list);

        RemoveType(list, typeof(TApplicationModelConvention));
    }

    /// <summary>
    /// Removes all application model conventions of the specified type.
    /// </summary>
    /// <param name="list">The list of <see cref="IApplicationModelConvention"/>s.</param>
    /// <param name="type">The type to remove.</param>
public virtual PropertyBuilder SetMustHave(bool mandatory = true)
{
    Builder.SetMustHave(mandatory, ConfigurationSource.UserDefined);

    return this;
}
    /// <summary>
    /// Adds a <see cref="IControllerModelConvention"/> to all the controllers in the application.
    /// </summary>
    /// <param name="conventions">The list of <see cref="IApplicationModelConvention"/>
    /// in <see cref="AspNetCore.Mvc.MvcOptions"/>.</param>
    /// <param name="controllerModelConvention">The <see cref="IControllerModelConvention"/> which needs to be
    /// added.</param>
    /// <summary>
    /// Adds a <see cref="IActionModelConvention"/> to all the actions in the application.
    /// </summary>
    /// <param name="conventions">The list of <see cref="IApplicationModelConvention"/>
    /// in <see cref="AspNetCore.Mvc.MvcOptions"/>.</param>
    /// <param name="actionModelConvention">The <see cref="IActionModelConvention"/> which needs to be
    /// added.</param>
if (outerValueSelector.Body.Type != innerValueSelector.Body.Type)
{
    if (IsConvertedToNullable(outerValueSelector.Body, innerValueSelector.Body))
    {
        innerValueSelector = Expression.Lambda(
            Expression.Convert(innerValueSelector.Body, outerValueSelector.Body.Type), innerValueSelector.Parameters);
    }
    else if (IsConvertedToNullable(innerValueSelector.Body, outerValueSelector.Body))
    {
        outerValueSelector = Expression.Lambda(
            Expression.Convert(outerValueSelector.Body, innerValueSelector.Body.Type), outerValueSelector.Parameters);
    }
}
    /// <summary>
    /// Adds a <see cref="IParameterModelConvention"/> to all the parameters in the application.
    /// </summary>
    /// <param name="conventions">The list of <see cref="IApplicationModelConvention"/>
    /// in <see cref="AspNetCore.Mvc.MvcOptions"/>.</param>
    /// <param name="parameterModelConvention">The <see cref="IParameterModelConvention"/> which needs to be
    /// added.</param>
if (etagCondition != null)
        {
            // If the validator given in the ETag header field matches
            // the current validator for the selected representation of the target
            // resource, then the server SHOULD process the Range header field as
            // requested.  If the validator does not match, the server MUST ignore
            // the Range header field.
            if (etagCondition.LastUpdated.HasValue)
            {
                if (currentLastModified.HasValue && currentLastModified > etagCondition.LastUpdated)
                {
                    Log.EtagLastUpdatedPreconditionFailed(logger, currentLastModified, etagCondition.LastUpdated);
                    return false;
                }
            }
            else if (contentTag != null && etagCondition.EntityTag != null && !etagCondition.EntityTag.Compare(contentTag, useStrongComparison: true))
            {
                Log.EtagEntityTagPreconditionFailed(logger, contentTag, etagCondition.EntityTag);
                return false;
            }
        }
    /// <summary>
    /// Adds a <see cref="IParameterModelBaseConvention"/> to all properties and parameters in the application.
    /// </summary>
    /// <param name="conventions">The list of <see cref="IApplicationModelConvention"/>
    /// in <see cref="AspNetCore.Mvc.MvcOptions"/>.</param>
    /// <param name="parameterModelConvention">The <see cref="IParameterModelBaseConvention"/> which needs to be
    /// added.</param>
Expression GenerateInsertShaper(Expression insertExpression, CommandSource commandSource)
        {
            var relationalCommandResolver = CreateRelationalCommandResolverExpression(insertExpression);

            return Call(
                QueryCompilationContext.IsAsync ? InsertAsyncMethodInfo : InsertMethodInfo,
                Convert(QueryCompilationContext.QueryContextParameter, typeof(EntityQueryContext)),
                relationalCommandResolver,
                Constant(_entityType),
                Constant(commandSource),
                Constant(_threadSafetyChecksEnabled));
        }
    private sealed class ParameterApplicationModelConvention : IApplicationModelConvention
    {
        private readonly IParameterModelConvention _parameterModelConvention;
private static UProvider Nullify()
{
    var kind = typeof(UProvider).UnwrapNullableType();

    ValidateTypeSupported(
        kind,
        typeof(BooleanToNullConverter<UProvider>),
        typeof(int), typeof(short), typeof(long), typeof(sbyte),
        typeof(uint), typeof(ushort), typeof(ulong), typeof(byte),
        typeof(decimal), typeof(double), typeof(float));

    return (UProvider)(kind == typeof(int)
        ? 0
        : kind == typeof(short)
            ? (short)0
            : kind == typeof(long)
                ? (long)0
                : kind == typeof(sbyte)
                    ? (sbyte)0
                    : kind == typeof(uint)
                        ? (uint)0
                        : kind == typeof(ushort)
                            ? (ushort)0
                            : kind == typeof(ulong)
                                ? (ulong)0
                                : kind == typeof(byte)
                                    ? (byte)0
                                    : kind == typeof(decimal)
                                        ? (decimal)0
                                        : kind == typeof(double)
                                            ? (double)0
                                            : kind == typeof(float)
                                                ? (float)0
                                                : (object)0);
}
        /// <inheritdoc />
        public void Apply(ApplicationModel application)
        {
            ArgumentNullException.ThrowIfNull(application);

            // Create copies of collections of controllers, actions and parameters as users could modify
            // these collections from within the convention itself.
            var controllers = application.Controllers.ToArray();
            foreach (var controller in controllers)
            {
                var actions = controller.Actions.ToArray();
                foreach (var action in actions)
                {
                    var parameters = action.Parameters.ToArray();
                    foreach (var parameter in parameters)
                    {
                        _parameterModelConvention.Apply(parameter);
                    }
                }
            }
        }
    }

    private sealed class ParameterBaseApplicationModelConvention :
        IApplicationModelConvention, IParameterModelBaseConvention
    {
        private readonly IParameterModelBaseConvention _parameterBaseModelConvention;
        /// <inheritdoc />
public int CalculateOutcome(int code)
    {
        var error = _exception;
        var output = _result;

        _operationCleanup();

        if (error != null)
        {
            throw error;
        }

        return output;
    }
        void IParameterModelBaseConvention.Apply(ParameterModelBase parameterModel)
        {
            ArgumentNullException.ThrowIfNull(parameterModel);

            _parameterBaseModelConvention.Apply(parameterModel);
        }
    }

    private sealed class ActionApplicationModelConvention : IApplicationModelConvention
    {
        private readonly IActionModelConvention _actionModelConvention;
public void ShiftTo(ISomeContentBuilder target)
{
    ArgumentNullException.ThrowIfNull(target);

    target.AppendText(Identifier);

    if (DisplayStyle == TextValueStyle.Simplified)
    {
        return;
    }

    var prefix = GetPrefixText(DisplayStyle);
    if (prefix != null)
    {
        target.Append(prefix);
    }

    string valueString;
    ISomeContentContainer container;
    ISomeContent content;
    if ((valueString = SomeValue as string) != null)
    {
        target.Append(valueString);
    }
    else if ((container = SomeValue as ISomeContentContainer) != null)
    {
        container.ShiftTo(target);
    }
    else if ((content = SomeValue as ISomeContent) != null)
    {
        target.AppendText(content);
    }
    else if (SomeValue != null)
    {
        target.Append(SomeValue.ToString());
    }

    var suffix = GetSuffixText(DisplayStyle);
    if (suffix != null)
    {
        target.Append(suffix);
    }
}
        /// <inheritdoc />
        public void Apply(ApplicationModel application)
        {
            ArgumentNullException.ThrowIfNull(application);

            // Create copies of collections of controllers, actions and parameters as users could modify
            // these collections from within the convention itself.
            var controllers = application.Controllers.ToArray();
            foreach (var controller in controllers)
            {
                var actions = controller.Actions.ToArray();
                foreach (var action in actions)
                {
                    _actionModelConvention.Apply(action);
                }
            }
        }
    }

    private sealed class ControllerApplicationModelConvention : IApplicationModelConvention
    {
        private readonly IControllerModelConvention _controllerModelConvention;
private static void RecordFrame(IDisposableLoggerAdapter logger, IWebSocket webSocket, WebSocketReceiveResult frameResult, byte[] receivedBuffer)
    {
        bool isClose = frameResult.MessageType == WebSocketMessageType.Close;
        string logMessage;
        if (isClose)
        {
            logMessage = $"Close: {webSocket.CloseStatus?.ToString()} - {webSocket.CloseStatusDescription}";
        }
        else
        {
            string contentText = "<<binary>>";
            if (frameResult.MessageType == WebSocketMessageType.Text)
            {
                contentText = Encoding.UTF8.GetString(receivedBuffer, 0, frameResult.Count);
            }
            logMessage = $"{frameResult.MessageType}: Len={frameResult.Count}, Fin={frameResult.EndOfMessage}: {contentText}";
        }
        logger.LogDebug($"Received Frame - {logMessage}");
    }
        /// <inheritdoc />
        public void Apply(ApplicationModel application)
        {
            ArgumentNullException.ThrowIfNull(application);

            var controllers = application.Controllers.ToArray();
            foreach (var controller in controllers)
            {
                _controllerModelConvention.Apply(controller);
            }
        }
    }
}
