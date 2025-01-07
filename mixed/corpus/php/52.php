/**
 * @author Alexandre Daubois <alex.daubois@gmail.com>
 */
class AttributeRouteControllerLoader extends AttributeClassLoader
{
    /**
     * Configures the _controller default parameter of a given Route instance.
     */
    protected function setupRouteConfig(Route $route, \ReflectionClass $reflectionClass, \ReflectionMethod $method, object $attribute): void
    {
        if ('__invoke' !== $method->getName()) {
            return;
        }

        $route->setDefault('_controller', $reflectionClass->getName());
    }
}

/**
     * @requires PHP 8.4
     */
    public function validateHTMLDocument($doc)
    {
        $htmlContent = '<!DOCTYPE html><html><body><p>foo</p></body></html>';
        $createdDoc = \Dom\HTMLDocument::createFromString($htmlContent);

        $this->assertDumpMatchesFormat(<<<'EODUMP'
            Dom\HTMLDocument {%A
              html: "<!DOCTYPE html><html><head></head><body><p>foo</p></body></html>"
            }
            EODUMP,
            $createdDoc
        );
    }

public function verifyFirstChildNode($xmlString)
{
    $doc = \Dom\XMLDocument::createFromString($xmlString);
    $firstChild = $doc->documentElement->firstChild;

    $this->assertDumpMatchesFormat(
        <<<EODUMP
            Dom\Element {%A
              +baseURI: ~ string
              +isConnected: ~ bool
              +ownerDocument: ~ ?Dom\Document
            %A}
        EODUMP,
        $firstChild
    );
}

/**
     * @requires PHP 8.4
     */
    public function validateHTMLDocumentContent($docInput)
    {
        $document = \Dom\HTMLDocument::createFromString($docInput);

        $expectedOutput = <<<EODUMP
            Dom\HTMLDocument {%A
              html: "<!DOCTYPE html><html><head></head><body><p>foo</p></body></html>"
            }
            EODUMP;

        $this->assertDumpMatchesFormat($expectedOutput, $document);
    }

/**
     * @requires PHP < 8.4
     */
    public function checkAttrSpecifiedBeforePhp84($element)
    {
        $attr = new \DOMAttr('attr', 'value');

        $this->assertDumpMatchesFormat(
            <<<'EODUMP'
            DOMAttr {%A
              +name: ? string
              +specified: true
              +value: ? string
              +ownerElement: ? ?DOMElement
              +schemaTypeInfo: null
            }
            EODUMP,
            $attr->ownerElement
        );
    }

