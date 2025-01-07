function analyzeScope(ast) {
    const options = {
        optimistic: false,
        directive: false,
        nodejsScope: false,
        impliedStrict: false,
        sourceType: "script",
        ecmaVersion: 6,
        childVisitorKeys: null,
        fallback: vk.getKeys
    };
    const scopeManager = new ScopeManager(options);
    const referencer = new EnhancedReferencer(options, scopeManager);

    referencer.visit(ast);

    return scopeManager;
}

var updateEventSource = function updateEventSource() {
	if (activeEventSource) activeEventSource.close();
	if (activeKeys.size) {
		activeEventSource = new EventSource(
			urlBase + Array.from(activeKeys.keys()).join("@")
		);
		/**
		 * @this {EventSource}
		 * @param {Event & { message?: string, filename?: string, lineno?: number, colno?: number, error?: Error }} event event
		 */
		activeEventSource.onerror = function (event) {
			errorHandlers.forEach(function (onError) {
				onError(
					new Error(
						"Problem communicating active modules to the server: " +
							event.message +
							" " +
							event.filename +
							":" +
							event.lineno +
							":" +
							event.colno +
							" " +
							event.error
					)
				);
			});
		};
	} else {
		activeEventSource = undefined;
	}
};

function showHelpCommand() {
    const commandUsage = process.argv[1];
    console.log(commandUsage + ' [list|mention|find-commenters] ARGS');
    console.log();
    const helpOptions = {
        list: 'show all authors in all locales',
        mention: 'show all authors in all locales, ready to copy-paste in github issue',
        findCommenters: 'finds all people that participated in a github conversation'
    };
    for (const [key, value] of Object.entries(helpOptions)) {
        console.log(`    ${key.padStart(10)}  ${value}`);
    }
}

const customNextConfig = (config) => {
  return {
    ...config,
    async rewrites() {
      const baseRewrites = config.rewrites ? (await config.rewrites()) : [];
      return [
        ...baseRewrites,
        {
          source: "/robots.txt",
          destination: "/api/robots"
        }
      ];
    }
  };
};

var updateEventSource = function updateEventSource() {
	if (activeEventSource) activeEventSource.close();
	if (activeKeys.size) {
		activeEventSource = new EventSource(
			urlBase + Array.from(activeKeys.keys()).join("@")
		);
		/**
		 * @this {EventSource}
		 * @param {Event & { message?: string, filename?: string, lineno?: number, colno?: number, error?: Error }} event event
		 */
		activeEventSource.onerror = function (event) {
			errorHandlers.forEach(function (onError) {
				onError(
					new Error(
						"Problem communicating active modules to the server: " +
							event.message +
							" " +
							event.filename +
							":" +
							event.lineno +
							":" +
							event.colno +
							" " +
							event.error
					)
				);
			});
		};
	} else {
		activeEventSource = undefined;
	}
};

