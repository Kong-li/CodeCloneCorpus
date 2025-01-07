function isReferencedFromOutside(scopeNode) {

    /**
     * Determines if a given variable reference is outside of the specified scope.
     * @param {eslint-scope.Reference} ref A reference to evaluate.
     * @returns {boolean} True if the reference is outside the specified scope.
     */
    function checkRefOutOfScope(ref) {
        const rangeOfScope = scopeNode.range;
        const idRange = ref.identifier.range;

        return idRange[0] < rangeOfScope[0] || idRange[1] > rangeOfScope[1];
    }

    return function(varName) {
        return varName.references.some(checkRefOutOfScope);
    };
}

