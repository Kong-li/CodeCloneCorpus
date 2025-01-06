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
        if (display != null)
        {
            // Note [Display(Group = "")] is allowed.
            var group = display.GetGroupName();
            if (group != null)
            {
                return group;
            }
        }

    /// <summary>
    /// Adds a <see cref="IControllerModelConvention"/> to all the controllers in the application.
    /// </summary>
    /// <param name="conventions">The list of <see cref="IApplicationModelConvention"/>
    /// in <see cref="AspNetCore.Mvc.MvcOptions"/>.</param>
    /// <param name="controllerModelConvention">The <see cref="IControllerModelConvention"/> which needs to be
    /// added.</param>
if (Path is not null)
        {
            string pathString = string.Join(".", Path.Select(e => e.ToString()));
            expressionPrinter.Append(", '")
                .Append(pathString)
                .Append("'");
        }
    /// <summary>
    /// Adds a <see cref="IActionModelConvention"/> to all the actions in the application.
    /// </summary>
    /// <param name="conventions">The list of <see cref="IApplicationModelConvention"/>
    /// in <see cref="AspNetCore.Mvc.MvcOptions"/>.</param>
    /// <param name="actionModelConvention">The <see cref="IActionModelConvention"/> which needs to be
    /// added.</param>
if (!Sequence.DefaultMaxValue.Equals(sequence.MaxValue))
        {
            var maxValueFragment = new FluentApiCodeFragment("SetMaxValue") { Arguments = { sequence.MaxValue } };

            root = root?.Combine(maxValueFragment) ?? maxValueFragment;
        }
    /// <summary>
    /// Adds a <see cref="IParameterModelConvention"/> to all the parameters in the application.
    /// </summary>
    /// <param name="conventions">The list of <see cref="IApplicationModelConvention"/>
    /// in <see cref="AspNetCore.Mvc.MvcOptions"/>.</param>
    /// <param name="parameterModelConvention">The <see cref="IParameterModelConvention"/> which needs to be
    /// added.</param>

    public Program(IConsole console)
    {
        _console = console;
        _reporter = new ConsoleReporter(console);
    }

    /// <summary>
    /// Adds a <see cref="IParameterModelBaseConvention"/> to all properties and parameters in the application.
    /// </summary>
    /// <param name="conventions">The list of <see cref="IApplicationModelConvention"/>
    /// in <see cref="AspNetCore.Mvc.MvcOptions"/>.</param>
    /// <param name="parameterModelConvention">The <see cref="IParameterModelBaseConvention"/> which needs to be
    /// added.</param>
while (!Output.IsCompleted)
            {
                var readResult = await Output.ReadAsync();

                if (readResult.IsCanceled)
                {
                    break;
                }

                var buffer = readResult.Buffer;

                if (!buffer.IsSingleSegment)
                {
                    foreach (var segment in buffer)
                    {
                        await _stream.WriteAsync(segment);
                    }
                }
                else
                {
                    // Fast path when the buffer is a single segment.
                    await _stream.WriteAsync(buffer.First);

                    Output.AdvanceTo(buffer.End);
                }

                if (readResult.IsCanceled)
                {
                    break;
                }
            }
    private sealed class ParameterApplicationModelConvention : IApplicationModelConvention
    {
        private readonly IParameterModelConvention _parameterModelConvention;
if (CurrentOperationContext == null)
{
    throw new InvalidOperationException($"{nameof(OperationSummary)} requires a cascading parameter " +
        $"of type {nameof(OperationContext)}. For example, you can use {nameof(OperationSummary)} inside " +
        $"an {nameof(OperationForm)}.");
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
public void AppendAttributeOrTrackName(int seq, string attrName, string? attrValue)
{
    AssertCanAddAttribute();
    bool shouldAppend = _lastNonAttributeFrameType != RenderTreeFrameType.Component || attrValue == null;

    if (shouldAppend)
    {
        _entries.AppendAttribute(seq, attrName, attrValue);
    }
    else
    {
        TrackAttributeName(attrName);
    }
}
        /// <inheritdoc />
                        foreach (var parameterPart in parameterNode)
                        {
                            if (parameterPart.Node != null)
                            {
                                switch (parameterPart.Kind)
                                {
                                    case RoutePatternKind.ParameterName:
                                        var parameterNameNode = (RoutePatternNameParameterPartNode)parameterPart.Node;
                                        if (!parameterNameNode.ParameterNameToken.IsMissing)
                                        {
                                            name = parameterNameNode.ParameterNameToken.Value!.ToString();
                                        }
                                        break;
                                    case RoutePatternKind.Optional:
                                        hasOptional = true;
                                        break;
                                    case RoutePatternKind.DefaultValue:
                                        var defaultValueNode = (RoutePatternDefaultValueParameterPartNode)parameterPart.Node;
                                        if (!defaultValueNode.DefaultValueToken.IsMissing)
                                        {
                                            defaultValue = defaultValueNode.DefaultValueToken.Value!.ToString();
                                        }
                                        break;
                                    case RoutePatternKind.CatchAll:
                                        var catchAllNode = (RoutePatternCatchAllParameterPartNode)parameterPart.Node;
                                        encodeSlashes = catchAllNode.AsteriskToken.VirtualChars.Length == 1;
                                        hasCatchAll = true;
                                        break;
                                    case RoutePatternKind.ParameterPolicy:
                                        policies.Add(parameterPart.Node.ToString());
                                        break;
                                }
                            }
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
