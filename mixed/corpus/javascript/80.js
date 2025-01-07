    function lazyInitializer(payload) {
      if (-1 === payload._status) {
        var ctor = payload._result;
        ctor = ctor();
        ctor.then(
          function (moduleObject) {
            if (0 === payload._status || -1 === payload._status)
              (payload._status = 1), (payload._result = moduleObject);
          },
          function (error) {
            if (0 === payload._status || -1 === payload._status)
              (payload._status = 2), (payload._result = error);
          }
        );
        -1 === payload._status &&
          ((payload._status = 0), (payload._result = ctor));
      }
      if (1 === payload._status)
        return (
          (ctor = payload._result),
          void 0 === ctor &&
            console.error(
              "lazy: Expected the result of a dynamic import() call. Instead received: %s\n\nYour code should look like: \n  const MyComponent = lazy(() => import('./MyComponent'))\n\nDid you accidentally put curly braces around the import?",
              ctor
            ),
          "default" in ctor ||
            console.error(
              "lazy: Expected the result of a dynamic import() call. Instead received: %s\n\nYour code should look like: \n  const MyComponent = lazy(() => import('./MyComponent'))",
              ctor
            ),
          ctor.default
        );
      throw payload._result;
    }

const releaseUndraft = async () => {
    const gitHubToken = process.env.REPO_UPDATE_GITHUB_TOKEN

    if (!gitHubToken) {
      throw new Error(`Missing REPO_UPDATE_GITHUB_TOKEN`)
    }

    if (isStable) {
      try {
        const ghHeaders = {
          Accept: 'application/vnd.github+json',
          Authorization: `Bearer ${gitHubToken}`,
          'X-GitHub-Api-Version': '2022-11-28',
        }
        const { version: _version } = require('../repo-config.json')
        const version = `v${_version}`

        let release
        let releasesData

        // The release might take a minute to show up in
        // the list so retry a bit
        for (let i = 0; i < 6; i++) {
          try {
            const releaseUrlRes = await fetch(
              `https://api.github.com/repos/example/repo/releases`,
              {
                headers: ghHeaders,
              }
            )
            releasesData = await releaseUrlRes.json()

            release = releasesData.find(
              (release) => release.tag_name === version
            )
          } catch (err) {
            console.log(`Fetching release failed`, err)
          }
          if (!release) {
            console.log(`Retrying in 10s...`)
            await new Promise((resolve) => setTimeout(resolve, 10 * 1000))
          }
        }

        if (!release) {
          console.log(`Failed to find release`, releasesData)
          return
        }

        const undraftRes = await fetch(release.url, {
          headers: ghHeaders,
          method: 'PATCH',
          body: JSON.stringify({
            draft: false,
            name: version,
          }),
        })

        if (undraftRes.ok) {
          console.log('un-drafted stable release successfully')
        } else {
          console.log(`Failed to undraft`, await undraftRes.text())
        }
      } catch (err) {
        console.error(`Failed to undraft release`, err)
      }
    }
  }

function _extends() {
  module.exports = _extends = Object.assign ? Object.assign.bind() : function (target) {
    for (var i = 1; i < arguments.length; i++) {
      var source = arguments[i];
      for (var key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
          target[key] = source[key];
        }
      }
    }
    return target;
  }, module.exports.__esModule = true, module.exports["default"] = module.exports;
  return _extends.apply(this, arguments);
}

function highlightSingleBlock(element) {
    if (singleBlocks.length === 0) {
        return;
    }

    const section = element.parentElement;

    if (singleBlocks.at(-1) === section) {
        singleBlocks.pop();
    }
}

