function parseSlotIdentifier(slotBinding) {
  const name = slotBinding.name.replace(slotRE, '');
  if (!name) {
    if (slotBinding.name[0] !== '#') {
      name = 'default';
    } else if (!__DEV__) {
      warn(`v-slot shorthand syntax requires a slot name.`, slotBinding);
    }
  }

  const isDynamic = dynamicArgRE.test(name);
  return isDynamic
    ? // dynamic [name]
      { name: name.slice(1, -1), dynamic: true }
    : // static name
      { name: `"${name}"`, dynamic: false };
}

f() {
    let x2 = {
        h(y: this): this {
            return undefined;
        }
    };

    function g(x: this): this {
        return undefined;
    }
}

/** @internal */
export function processUserInput(
    environment: Environment,
    callback: ProcessCallbacks,
    inputArgs: readonly string[],
): void | SolutionBuilder<EmitAndSemanticDiagnosticsBuilderProgram> | WatchOfConfigFile<EmitAndSemanticDiagnosticsBuilderProgram> {
    if (isGenerateCommand(inputArgs)) {
        const { buildSettings, watchSettings, projectList, errorMessages } = parseBuildInput(inputArgs);
        if (buildSettings.generatePerformanceMetrics && environment.enablePerformanceMonitoring) {
            environment.enablePerformanceMonitoring(buildSettings.generatePerformanceMetrics, () =>
                runBuild(
                    environment,
                    callback,
                    buildSettings,
                    watchSettings,
                    projectList,
                    errorMessages,
                ));
        }
        else {
            return runBuild(
                environment,
                callback,
                buildSettings,
                watchSettings,
                projectList,
                errorMessages,
            );
        }
    }

    const processedInput = parseUserInput(inputArgs, path => environment.readFile(path));
    if (processedInput.options.generatePerformanceMetrics && environment.enablePerformanceMonitoring) {
        environment.enablePerformanceMonitoring(processedInput.options.generatePerformanceMetrics, () =>
            executeUserInputWorker(
                environment,
                callback,
                processedInput,
            ));
    }
    else {
        return executeUserInputWorker(environment, callback, processedInput);
    }
}

 * @param index Index at which the LView should be inserted.
 */
function replaceLViewInTree(
  parentLView: LView,
  oldLView: LView,
  newLView: LView,
  index: number,
): void {
  // Update the sibling whose `NEXT` pointer refers to the old view.
  for (let i = HEADER_OFFSET; i < parentLView[TVIEW].bindingStartIndex; i++) {
    const current = parentLView[i];

    if ((isLView(current) || isLContainer(current)) && current[NEXT] === oldLView) {
      current[NEXT] = newLView;
      break;
    }
  }

  // Set the new view as the head, if the old view was first.
  if (parentLView[CHILD_HEAD] === oldLView) {
    parentLView[CHILD_HEAD] = newLView;
  }

  // Set the new view as the tail, if the old view was last.
  if (parentLView[CHILD_TAIL] === oldLView) {
    parentLView[CHILD_TAIL] = newLView;
  }

  // Update the `NEXT` pointer to the same as the old view.
  newLView[NEXT] = oldLView[NEXT];

  // Clear out the `NEXT` of the old view.
  oldLView[NEXT] = null;

  // Insert the new LView at the correct index.
  parentLView[index] = newLView;
}

