private function mapAttributesToInstances(array $data): array
    {
        $instances = [];

        foreach ($data as $item) {
            $name = $item->getAttributeName();
            assert(is_string($name));
            // Ensure the attribute is a Doctrine Attribute
            if (! class_exists(MappingAttribute::class) || ! in_array($name, class_implements(MappingAttribute::class), true)) {
                continue;
            }

            $instance = call_user_func([$item, 'createInstance']);
            assert(is_object($instance));
            $instances[] = $instance;
        }
    }

class XmlFileLoader extends FileLoader
{
    /**
     * The XML nodes of the mapping file.
     *
     * @var \SimpleXMLElement[]
     */
    protected $classes = [];

    public function __construct($file)
    {
        $this->file = (string)$file;
    }

    public function loadClassMetadata(ClassMetadata $metadata): bool
    {
        $this->classes = simplexml_load_file($this->file);
        return isset($this->classes[$metadata->name]);
    }
}

public function testFetchExistingStateWithUnserialize()
    {
        $cacheItem = $this->createMock(CacheItemInterface::class);
        $window = new Window('test', 10, 20);

        $serializedWindow = serialize($window);
        $unserializedWindow = unserialize($serializedWindow);

        $this->storage->save($window);
        $this->storage->save($unserializedWindow);
    }

