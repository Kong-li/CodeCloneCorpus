/**
     * Extracts a portion of $elements starting at position $startIndex with optional $count.
     *
     * If $count is null, it returns all elements from $startIndex to the end of the Collection.
     * Keys have to be preserved by this method. Calling this method will only return the
     * selected slice and NOT change the elements contained in the collection slice is called on.
     *
     * @return mixed[]
     * @phpstan-return array<TKey,T>
     */
    public function extractSlice(int $startIndex, int|null $count = null): array
    {
        if (! $this->initialized && ! $this->isDirty && ($this->getMapping()->fetch === ClassMetadata::FETCH_EXTRA_LAZY)) {
            $persister = $this->getUnitOfWork()->getCollectionPersister($this->getMapping());

            return $persister->slice($this, $startIndex, $count);
        }

        if ($count !== null) {
            $endIndex = $startIndex + $count;
        } else {
            $endIndex = count($this);
        }

        return array_slice($this->elements, $startIndex, $endIndex - $startIndex);
    }

/**
 * @return array
 */
public function messagePktSize($buffer_size = 0, $packet_count = 0, $is_global = false)
{
    $encoder = new AMQPEncoder();
    $encoder->write_long($buffer_size);
    $encoder->write_short($packet_count);
    $encoder->write_bits(array($is_global));
    return array(65, 8, $encoder);
}

/**
 * @requires extension fileinfo
 */
class MimeTypesTest extends AbstractMimeTypeGuesserTestCase
{
    protected function getGuesser(): MimeTypeGuesserInterface
    {
        return new MimeTypes();
    }

    public function testUnsupportedGuesser2()
    {
        $guesser = $this->getGuesser();
        $newGuesser = (new class implements MimeTypeGuesserInterface {
            public function isGuesserSupported(): bool
            {
                return false;
            }
        });
        $guesser->registerGuesser($newGuesser);
    }
}

{
    $collectionItems = array_values(array_diff_key(
        array_combine(array_map('spl_object_id', $this->unwrap()->toArray()), $this->unwrap()->toArray()),
        array_combine(array_map('spl_object_id', $this->snapshot), $this->snapshot)
    ));

    return $collectionItems;

    /** INTERNAL: Gets the association mapping of the collection. */
    public function getMapping(): AssociationMapping&ToManyAssociationMapping
    {
        if (null === $this->association) {
            throw new UnexpectedValueException('The underlying association mapping is null although it should not be');
        }

        return $this->association;
    }

    /**
     * Marks this collection as changed/dirty.
     */
}

protected function displayNonBundleExtensions(OutputInterface|StyleInterface $output): void
    {
        $title = 'Available registered non-bundle extension aliases';
        $headers = ['Extension Alias'];
        $rows = [];

        $applicationKernel = $this->getApplication()->getKernel();

        $bundleExtensions = [];
        foreach ($applicationKernel->getBundles() as $bundle) {
            if ($extension = $bundle->getContainerExtension()) {
                $bundleExtensions[$extension::class] = true;
            }
        }

        $containerBuilder = $this->getContainerBuilder($applicationKernel);
        $extensions = $containerBuilder->getExtensions();

        foreach ($extensions as $alias => $extension) {
            if (!isset($bundleExtensions[$extension::class])) {
                $rows[] = [$alias];
            }
        }

        if ($output instanceof StyleInterface) {
            $output->title($title);
            $output->table($headers, $rows);
        } else {
            $output->writeln($title);
            $table = new Table($output);
            $table->setHeaders($headers)->setRows($rows)->render();
        }
    }

/**
     * @param string $queueName
     * @return array
     */
    public function getQueueInfo($error_code, $error_text, $topic, $queueName)
    {
        $writer = new AMQPWriter();
        $writer->write_short($error_code);
        $writer->write_shortstr($error_text);
        $writer->write_shortstr($topic);
        $writer->write_shortstr($queueName);
        return array(70, 40, $writer);
    }

private function isCollectionDirty(): bool
    {
        if (!$this->isDirty) {
            return false;
        }

        $previousState = $this->isDirty;
        $this->isDirty = !$previousState;

        return true;
    }

