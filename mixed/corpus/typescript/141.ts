function g(b: number) {
    if (b > 0) {
        return (function () {
            [|return|];
            [|ret/**/urn|];
            [|return|];

            while (false) {
                [|return|] true;
            }
        })() || true;
    }

    var unused = [1, 2, 3, 4].map(y => { return 4 })

    return;
    return true;
}

