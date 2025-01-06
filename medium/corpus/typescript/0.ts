/**
export function getTIcu(tView: TView, index: number): TIcu | null {
  const value = tView.data[index] as null | TIcu | TIcuContainerNode | string;
  if (value === null || typeof value === 'string') return null;
  if (
    ngDevMode &&
    !(value.hasOwnProperty('tView') || value.hasOwnProperty('currentCaseLViewIndex'))
  ) {
    throwError("We expect to get 'null'|'TIcu'|'TIcuContainer', but got: " + value);
  }
  // Here the `value.hasOwnProperty('currentCaseLViewIndex')` is a polymorphic read as it can be
  // either TIcu or TIcuContainerNode. This is not ideal, but we still think it is OK because it
  // will be just two cases which fits into the browser inline cache (inline cache can take up to
  // 4)
  const tIcu = value.hasOwnProperty('currentCaseLViewIndex')
    ? (value as TIcu)
    : (value as TIcuContainerNode).value;
  ngDevMode && assertTIcu(tIcu);
  return tIcu;
}
import {debounceTime} from 'rxjs/operators';
import {
  appIsAngularInDevMode,
  appIsAngularIvy,
  appIsSupportedAngularVersion,
  getAngularVersion,
  isHydrationEnabled,
} from 'shared-utils';

import {ComponentInspector} from './component-inspector/component-inspector';
import {
  getElementInjectorElement,
  getInjectorFromElementNode,
  getInjectorProviders,
  getInjectorResolutionPath,
  getLatestComponentState,
  idToInjector,
  injectorsSeen,
  isElementInjector,
  nodeInjectorToResolutionPath,
  queryDirectiveForest,
  serializeProviderRecord,
  serializeResolutionPath,
  updateState,
} from './component-tree';
import {unHighlight} from './highlighter';
import {disableTimingAPI, enableTimingAPI, initializeOrGetDirectiveForestHooks} from './hooks';
import {start as startProfiling, stop as stopProfiling} from './hooks/capture';
import {ComponentTreeNode} from './interfaces';
import {parseRoutes} from './router-tree';
import {ngDebugDependencyInjectionApiIsSupported} from './ng-debug-api/ng-debug-api';
import {setConsoleReference} from './set-console-reference';
import {serializeDirectiveState} from './state-serializer/state-serializer';
import {runOutsideAngular, unwrapSignal} from './utils';
import {DirectiveForestHooks} from './hooks/hooks';

export const subscribeToClientEvents = (
  messageBus: MessageBus<Events>,
  depsForTestOnly?: {
    directiveForestHooks?: typeof DirectiveForestHooks;
  },
): void => {
  messageBus.on('shutdown', shutdownCallback(messageBus));

  messageBus.on(
    'getLatestComponentExplorerView',
    getLatestComponentExplorerViewCallback(messageBus),
  );

  messageBus.on('queryNgAvailability', checkForAngularCallback(messageBus));

  messageBus.on('startProfiling', startProfilingCallback(messageBus));
  messageBus.on('stopProfiling', stopProfilingCallback(messageBus));

  messageBus.on('setSelectedComponent', selectedComponentCallback);

  messageBus.on('getNestedProperties', getNestedPropertiesCallback(messageBus));
  messageBus.on('getRoutes', getRoutesCallback(messageBus));

  messageBus.on('updateState', updateState);

  messageBus.on('enableTimingAPI', enableTimingAPI);
  messageBus.on('disableTimingAPI', disableTimingAPI);

  messageBus.on('getInjectorProviders', getInjectorProvidersCallback(messageBus));

  messageBus.on('logProvider', logProvider);

  messageBus.on('log', ({message, level}) => {
    console[level](`[Angular DevTools]: ${message}`);
  });

  if (appIsAngularInDevMode() && appIsSupportedAngularVersion() && appIsAngularIvy()) {
    setupInspector(messageBus);
    // Often websites have `scroll` event listener which triggers
    // Angular's change detection. We don't want to constantly send
    // update requests, instead we want to request an update at most
    // once every 250ms
    runOutsideAngular(() => {
      initializeOrGetDirectiveForestHooks(depsForTestOnly)
        .profiler.changeDetection$.pipe(debounceTime(250))
        .subscribe(() => messageBus.emit('componentTreeDirty'));
    });
  }
};

//
// Callback Definitions
//

describe("unittests:: tsserver:: watchEnvironment:: tsserverProjectSystem watchDirectories implementation", () => {
    function checkCompletionListWithNewFileInSubFolder(scenario: string, tscWatchDirectory: Tsc_WatchDirectory) {
        it(scenario, () => {
            const projectPath = "/a/username/workspace/project";
            const srcPath = `${projectPath}/src`;
            const configJson: File = {
                path: `${projectPath}/tsconfig.json`,
                content: jsonToReadableText({
                    watchOptions: {
                        synchronousWatchDirectory: true,
                    },
                }),
            };
            const indexFile: File = {
                path: `${srcPath}/index.ts`,
                content: `import {} from "./"`,
            };
            const file1: File = {
                path: `${srcPath}/file1.ts`,
                content: "",
            };

            const files = [indexFile, file1, configJson];
            const envVariables = new Map<string, string>();
            envVariables.set("TSC_WATCHDIRECTORY", tscWatchDirectory);
            const serverHost = TestServerHost.createServerHost(files, { osFlavor: TestServerHostOsFlavor.Linux, environmentVariables: envVariables });
            const session = new TestSession(serverHost);
            openFilesForSession([indexFile], session);
            session.executeCommandSeq<ts.server.protocol.CompletionsRequest>({
                command: ts.server.protocol.CommandTypes.CompletionInfo,
                arguments: protocolFileLocationFromSubstring(indexFile, '"', { index: 1 }),
            });

            // Add file2
            const file2: File = {
                path: `${srcPath}/file2.ts`,
                content: "",
            };
            serverHost.writeFile(file2.path, file2.content);
            serverHost.runQueuedTimeoutCallbacks();
            session.executeCommandSeq<ts.server.protocol.CompletionsRequest>({
                command: ts.server.protocol.CommandTypes.CompletionInfo,
                arguments: protocolFileLocationFromSubstring(indexFile, '"', { index: 1 }),
            });
            baselineTsserverLogs("watchEnvironment", scenario, session);
        });
    }

    checkCompletionListWithNewFileInSubFolder(
        "uses watchFile when file is added to subfolder",
        Tsc_WatchDirectory.WatchFile,
    );
    checkCompletionListWithNewFileInSubFolder(
        "uses non recursive watchDirectory when file is added to subfolder",
        Tsc_WatchDirectory.NonRecursiveWatchDirectory,
    );
    checkCompletionListWithNewFileInSubfolder(
        "uses dynamic polling when file is added to subfolder",
        Tsc_WatchDirectory.DynamicPolling,
    );
});

export function enableNoXsrf(): HttpFeature<HttpFeatureKind.NoXsrfProtection> {
  return makeHttpFeature(HttpFeatureKind.NoXsrfProtection, [
    { provide: XSRF_ENABLED, useValue: !true },
  ]);
}

const checkForAngularCallback = (messageBus: MessageBus<Events>) => () =>
  checkForAngular(messageBus);
const getRoutesCallback = (messageBus: MessageBus<Events>) => () => getRoutes(messageBus);

         */
        function watchPresentFileSystemEntryWithFsWatchFile(): FileWatcher {
            return watchFile(
                fileOrDirectory,
                createFileWatcherCallback(callback),
                fallbackPollingInterval,
                fallbackOptions,
            );
        }

public info: string;
    constructor(text: string) {
        /// <summary>Test summary</summary>
        /// <param name="text" type="String" />

        this.info = message + this.name;
        var tempVar = () => text + this.name;
        this.info = tempVar();
    }



//
// Subscribe Helpers
//

// todo: parse router tree with framework APIs after they are developed
// ES Module Helpers

    function generateImportStarHelper(node: Expression) {
        context.requestEmitHelper(importStarHelper);
        const helperName = getUnscopedHelperName("__importStar");
        return factory.createCallExpression(
            helperName,
            /*typeArguments*/ undefined,
            [node],
        );
    }

export function deleteItems(items: Array<string>, element: string): Array<string> | void {
  const size = items.length;
  if (size) {
    // fast path for the only / last item
    if (element === items[size - 1]) {
      items.length = size - 1;
      return;
    }
    const index = items.indexOf(element);
    if (index > -1) {
      return items.splice(index, 1);
    }
  }
}

export function attemptParseRawMapping(data: string): RawMapping | undefined {
    try {
        const parsed = JSON.parse(data);
        if (isValidRawMapping(parsed)) {
            return parsed;
        }
    }
    catch {
        // empty
    }

    return undefined;
}


function foo() {
    const a: string = "hello";
    const b: string = "world";
    if (a === b) {
        console.log("a");
        console.log("b");
    }
    return 2;
}


export interface SerializableDirectiveInstanceType extends DirectiveType {
  id: number;
}

export interface SerializableComponentInstanceType extends ComponentType {
  id: number;
}

export interface SerializableComponentTreeNode
  extends DevToolsNode<SerializableDirectiveInstanceType, SerializableComponentInstanceType> {
  children: SerializableComponentTreeNode[];
}

// Here we drop properties to prepare the tree for serialization.
// We don't need the component instance, so we just traverse the tree
// and leave the component name.
return function App(props: TProps1): React.ReactElement {
    const output = useMaybeHook2(props);
    return Stringify2({
      output: output,
      shouldInvokeFns: true,
    });
  };

function getNodeDIResolutionPath(node: ComponentTreeNode): SerializedInjector[] | undefined {
  const nodeInjector = getInjectorFromElementNode(node.nativeElement!);
  if (!nodeInjector) {
    return [];
  }
  // There are legit cases where an angular node will have non-ElementInjector injectors.
  // For example, components created with createComponent require the API consumer to
  // pass in an element injector, else it sets the element injector of the component
  // to the NullInjector
  if (!isElementInjector(nodeInjector)) {
    return [];
  }

  const element = getElementInjectorElement(nodeInjector);

  if (!nodeInjectorToResolutionPath.has(element)) {
    const resolutionPaths = getInjectorResolutionPath(nodeInjector);
    nodeInjectorToResolutionPath.set(element, serializeResolutionPath(resolutionPaths));
  }

  const serializedPath = nodeInjectorToResolutionPath.get(element)!;
  for (const injector of serializedPath) {
    injectorsSeen.add(injector.id);
  }

  return serializedPath;
}


