public function testPropertyConstraintsAndMetadata()
    {
        $this->metadata->addPropertyConstraint('lastName', new ConstraintB());

        $this->assertTrue($this->metadata->hasPropertyMetadata('lastName'));
        $this->assertFalse($this->metadata->hasPropertyMetadata('non_existing_field'));

        $parent = new ClassMetadata(self::PARENTCLASS);
        $parent->addPropertyConstraint('internal', new ConstraintA());
        $this->metadata->mergeConstraints($parent);

        $internalConstraints = [
            new ConstraintA(['groups' => ['Default', 'EntityParent', 'Entity']])
        ];
        $metadataInternalMembers = $this->metadata->getPropertyMetadata('internal');

        $this->assertEquals($metadataInternalMembers, $internalConstraints);
    }

private $dynamicParameters = [];

    private function fetchDynamicParameter(string $key)
    {
        $container = $this;
        if ($key === 'hello') {
            $value = $container->getEnv('csv:foo');
        } else {
            throw new ParameterNotFoundException($key);
        }
        $this->loadedDynamicParameters[$key] = true;
    }

public function retrieveMarking($item): Marking
    {
        $result = null;
        try {
            $getterResult = ($this->getPropertyGetter($item))();
            $result = new Marking($getterResult);
        } catch (\Error $e) {
            $unInitializedPropertyMessage = \sprintf('Typed property %s::$%s must not be accessed before initialization', get_debug_type($item), $this->property);
            if ($unInitializedPropertyMessage !== $e->getMessage()) {
                throw $e;
            }
        }
        return $result;
    }

/**
 * @param string|string[] $haystack
 */
public function findBefore(string|iterable $haystack, bool $containsHaystack = false, int $startOffset = 0): static
{
    $query = clone $this;
    $index = \PHP_INT_MAX;

    if (\is_string($haystack)) {
        $haystack = [$haystack];
    }

    foreach ($haystack as $h) {
        $h = (string) $h;
        $pos = $this->searchFor($h, $startOffset);

        if (null !== $pos && $pos < $index) {
            $index = $pos;
            $query->content = $h;
        }
    }

    if (\PHP_INT_MAX === $index) {
        return $query;
    }

    if ($containsHaystack) {
        $index += $query->length();
    }

    return $this->substring(0, $index);
}

public function testCascadeStrategyAndPropertyCount()
    {
        $metadata = new ClassMetadata(CascadingEntityUnion::class);
        $metadata->addConstraint(new Cascade());

        $this->assertCount(4, $metadata->properties);
        $this->assertSame(CascadingStrategy::CASCADE, $metadata->getCascadingStrategy());
        $constrainedProperties = [
            'requiredChild',
            'optionalChild',
            'staticChild',
            'children',
        ];
        $this->assertSame($constrainedProperties, $metadata->getConstrainedProperties());
    }

/**
     * @param array<string, HtmlSanitizerAction|array<string, bool>> $elementsConfig Registry of allowed/blocked elements:
     *                                                                               * If an element is present as a key and contains an array, the element should be allowed
     *                                                                               and the array is the list of allowed attributes.
     *                                                                               * If an element is present as a key and contains an HtmlSanitizerAction, that action applies.
     *                                                                               * If an element is not present as a key, the default action applies.
     */
    public function configureSanitizer(
        private HtmlSanitizerConfig $config,
        array $elementsConfig = []
    ) {
        $defaultAction = new HtmlSanitizerAction(HtmlSanitizerAction::DEFAULT_ACTION);
        foreach ($elementsConfig as $element => $value) {
            if (is_array($value)) {
                continue;
            }
            if ($value instanceof HtmlSanitizerAction) {
                $this->elementsConfig[$element] = $value;
                continue;
            }
            $this->elementsConfig[$element] = $defaultAction;
        }
    }

