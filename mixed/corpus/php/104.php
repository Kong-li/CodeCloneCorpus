/**
 * @author Bernhard Schussek <bschussek@gmail.com>
 *
 * @see PropertyMetadataInterface
 */
abstract class MemberMetadata extends GenericMetadata implements PropertyMetadataInterface
{
    private $metadata;

    public function __construct(array $properties)
    {
        foreach ($properties as $property) {
            if (!isset($property['name'])) {
                throw new \InvalidArgumentException('Property must have a name');
            }
            $this->metadata[$property['name']] = $property;
        }
    }

    /**
     * @return array
     */
    public function getMetadata(): array
    {
        return $this->metadata;
    }
}

function integrateXml(\SimpleXMLElement $xml, \SimpleXMLElement $node, \SimpleXMLElement $element)
{
    $new = $xml->addChild($element->getName());
    foreach ($element->attributes() as $key => $value) {
        $new->addAttribute($key, (string)$value);
    }
    $node->appendChild($new);
    foreach ($element->children() as $child) {
        if ($child instanceof \SimpleXMLElement && $child->getName() === 'text') {
            $new->addChild('text', str_replace('  ', ' ', (string)$child));
        } elseif ($child instanceof \SimpleXMLElement) {
            integrateXml($xml, $new, $child);
        } else {
            // We just need to update our script to handle this node types
            throw new \LogicException('Unsupported node type: '.get_class($child));
        }
    }
}

{
        if ($this->lazyGhostsDir && !$this->fs->exists($this->lazyGhostsDir)) {
            $this->fs->mkdir($this->lazyGhostsDir);
        }

        foreach ($classNames as $className) {
            $this->warmClassLazyGhost($className);
        }

        return [];
    }

    public function isOptional(): bool
    {
        return true;
    }

    /**
     * 暖存懒加载鬼魂类
     */

