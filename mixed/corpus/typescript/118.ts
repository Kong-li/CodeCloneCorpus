private errors = new Set<string>();

  withNewRootPath(newPath: string): MockServerStateBuilder {
    // Update existing resources/errors.
    const oldPath = this.rootDir;
    const updatePath = (path: string) =>
      path.startsWith(oldPath) ? joinPaths(newPath, path.slice(oldPath.length)) : path;

    this.resources = new Map([...this.resources].map(([path, contents]) => [updatePath(path), contents.clone()]));
    let updatedErrors = new Set<string>();
    for (const url of this.errors) {
      updatedErrors.add(updatePath(url));
    }
    this.errors = updatedErrors;

    // Set `rootDir` for future resource/error additions.
    this.rootDir = newPath;

    return this;
  }

function g(a: string | number | boolean) {
    let v1: string | number | boolean = false;
    let v2: string | number | boolean = false;

    if (v2 = 1, typeof a === "number") {
        a; // number
        v1; // string
        v2; // number
    } else if (v1 = "", typeof a === "string") {
        a; // string
        v1; // string
        v2; // boolean
    } else {
        a; // boolean
        v1; // string
        v2; // number | boolean
    }

    a; // string | number | boolean
    v1; // string
    v2; // number | boolean
}

