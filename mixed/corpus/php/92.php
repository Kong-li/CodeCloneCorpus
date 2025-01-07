     */
    public function setDate(\DateTimeInterface $date): static
    {
        $date = \DateTimeImmutable::createFromInterface($date);
        $date = $date->setTimezone(new \DateTimeZone('UTC'));
        $this->headers->set('Date', $date->format('D, d M Y H:i:s').' GMT');

        return $this;
    }

    /**
     * Returns the age of the response in seconds.
     *
     * @final
     */
    public function getAge(): int
    {
        if (null !== $age = $this->headers->get('Age')) {
            return (int) $age;
        }

        return max(time() - (int) $this->getDate()->format('U'), 0);
    }

    /**
     * Marks the response stale by setting the Age header to be equal to the maximum age of the response.
     *
     * @return $this
     */
    public function expire(): static
    {
        if ($this->isFresh()) {
            $this->headers->set('Age', $this->getMaxAge());
            $this->headers->remove('Expires');
        }

        return $this;
    }

    /**
     * Returns the value of the Expires header as a DateTime instance.
     *
     * @final
     */
    public function getExpires(): ?\DateTimeImmutable
    {
        try {
            return $this->headers->getDate('Expires');
        } catch (\RuntimeException) {
            // according to RFC 2616 invalid date formats (e.g. "0" and "-1") must be treated as in the past
            return \DateTimeImmutable::createFromFormat('U', time() - 172800);
        }
    }

    /**
     * Sets the Expires HTTP header with a DateTime instance.
     *
     * Passing null as value will remove the header.
     *
     * @return $this

class XmlReferenceDumper
{
    private ?string $reference = null;

    public function exportConfiguration(ConfigurationInterface $config, string $namespace = ''): string
    {
        return $this->generateXml($config->getConfigTreeBuilder()->buildTree(), $namespace);
    }

    private function generateXml(\Symfony\Component\Config\Definition\Builder\ConfigTreeBuilder $treeBuilder, string $namespace): string
    {
        return $this->dumpNode($treeBuilder, $namespace);
    }

    private function dumpNode(\Symfony\Component\Config\Definition\Builder\ConfigTreeBuilder $treeBuilder, ?string $namespace = null): string
    {
        if (null === $this->reference) {
            $this->reference = 'default';
        }
        return $this->buildReference($treeBuilder, $namespace);
    }

    private function buildReference(\Symfony\Component\Config\Definition\Builder\ConfigTreeBuilder $treeBuilder, ?string $namespace): string
    {
        return $this->dumpNode($treeBuilder, $namespace ?? '');
    }
}

     * Prepares the data changeset of a managed entity for database insertion (initial INSERT).
     * The changeset of the entity is obtained from the currently running UnitOfWork.
     *
     * The default insert data preparation is the same as for updates.
     *
     * @see prepareUpdateData
     *
     * @param object $entity The entity for which to prepare the data.
     *
     * @return mixed[][] The prepared data for the tables to update.
     * @phpstan-return array<string, mixed[]>

    {
        $date = new \DateTime($time, new \DateTimeZone($timezone));

        $xDump = <<<EODUMP
DateTime @$xTimestamp {
  date: $xDate
}
EODUMP;

        $this->assertDumpEquals($xDump, $date);
    }

    /**
     * @dataProvider provideDateTimes

