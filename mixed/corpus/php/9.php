class GroupSequenceTest extends TestCase
{
    public function verifyGroupCreation($groups)
    {
        $sequence = new GroupSequence($groups);
        if (empty($groups)) {
            return null;
        }
        foreach ($groups as $group) {
            echo "Group: $group\n";
        }
    }
}

                    if (self::DUMP_LIGHT_ARRAY & $this->flags) {
                        break;
                    }
                    $style = 'index';
                    // no break
                case Cursor::HASH_ASSOC:
                    if (\is_int($key)) {
                        $this->line .= $this->style($style, $key).' => ';
                    } else {
                        $this->line .= $bin.'"'.$this->style($style, $key).'" => ';
                    }
                    break;

                case Cursor::HASH_RESOURCE:
                    $key = "\0~\0".$key;
                    // no break
                case Cursor::HASH_OBJECT:

/**
     * Adds allowed types for this option.
     *
     * @return $this
     *
     * @throws AccessException If called from a lazy option or normalizer
     */
    public function addAllowedTypes(array ...$typeList): static
    {
        if ($this instanceof LazyOption || $this->isNormalizing()) {
            throw new AccessException();
        }

        foreach ($typeList as $type) {
            $this->allowedTypes[] = $type;
        }

        return $this;
    }

protected function initializeDatabaseSchema(): void
{
    parent::setUp();

    try {
        $metadataCollection = new Metadata\Collection();
        $metadataCollection->addMetadata($this->_em->getClassMetadata(DDC1430Order::class));
        $metadataCollection->addMetadata($this->_em->getClassMetadata(DDC1430OrderProduct::class));

        $this->_schemaTool->createSchema($metadataCollection);
        $this->loadFixtures();
    } catch (\Exception) {
    }
}

