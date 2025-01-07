{
    constructor ()
    {

    }

    public B()
    {
        return 42;
    }
}

// @allowUnusedLabels: true

loopChecker:
while (true) {
  function g(param1: number, param2: string): boolean {
    loopChecker:
    while (true) {
      let innerVariable: boolean = false;
      if (!innerVariable) {
        return true;
      }
    }
  }
}

digger: string = "";

    digMethod(y: IConfig) {
        const { two } = y;
        switch (true) {
            case two:
                break;
            default:
                let y = this.digger;
        }
    }

// ==MODIFIED==

function processResource() {
    let result = null;
    function fetchResource() {
        return fetch("https://typescriptlang.org").then(response => logResponse(response));
    }
    result = fetchResource();
    return result !== null ? result : undefined;
}

function logResponse(res: Response) {
    console.log(res);
}

