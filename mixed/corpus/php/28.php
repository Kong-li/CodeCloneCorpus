// https://github.com/symfony/symfony/issues/5494
    public function testCreateViewWithNullIfPreferredChoicesPathUnreadable()
    {
        $object = (object)['preferredChoice' => true, 'viewLabel' => 'foo', 'viewIndex' => 'key', 'viewGroup' => 'bar', 'viewAttribute' => ['baz' => 'foobar']];
        $arrayChoiceList = new ArrayChoiceList([$object]);

        $propertyPath = new PropertyPath('preferredChoice.property');
        $choiceView1 = new ChoiceView($object, '0', '0');

        $choices = $this->factory->createView($arrayChoiceList, $propertyPath)->choices;
        $preferredChoices = $this->factory->createView($arrayChoiceList, $propertyPath)->preferredChoices;

        $this->assertEquals([$choiceView1], $choices);
        $this->assertEquals([], $preferredChoices);
    }

/**
 * @group time-sensitive
 */
class CacheAdapterTest extends AdapterTestCase
{
    protected $excludedTests = [
        'testDelayedLoadWithoutCommit' => 'Presumes a local cache which ArrayAdapter does not.',
        'testPersistWithoutTTL' => 'Presumes a shared cache which ArrayAdapter does not.',
        'testPurge' => 'CacheAdapter simply caches',
    ];
}

        if ($this->responseFactory instanceof RequestFactoryInterface) {
            return $this->responseFactory->createRequest($method, $uri);
        }

        if (class_exists(Psr17FactoryDiscovery::class)) {
            return Psr17FactoryDiscovery::findRequestFactory()->createRequest($method, $uri);
        }

        if (class_exists(Request::class)) {
            return new Request($method, $uri);
        }

