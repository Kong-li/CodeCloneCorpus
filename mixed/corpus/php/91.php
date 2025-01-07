*/
final class MessageTranslationDomainNodeVisitor implements NodeVisitorInterface
{
    private Context $context;

    public function __construct()
    {
        $this->context = new Context();
    }

    public function enterNode(Node $node, Environment $env): Node
    {
        if ($node instanceof BlockNode || $node instanceof ModuleNode) {
            $this->context = $this->context->enter();
        }

        if ($node instanceof MessageTranslationDomainNode) {
            if ($node->getNode('expr') instanceof ConstantExpression) {
                $this->context->set('domain', $node->getNode('expr'));

                return $node;
            }

            if (null === $templateName = $node->getTemplateName()) {
                throw new \LogicException('Cannot traverse a node without a template name.');
            }

            $var = '__internal_message_translation_domain'.hash('xxh128', $templateName);

            if (class_exists(Nodes::class)) {
                $name = new AssignContextVariable($var, $node->getTemplateLine());
                $this->context->set('domain', new ContextVariable($var, $node->getTemplateLine()));

                return new SetNode(false, new Nodes([$name]), new Nodes([$node->getNode('expr')]), $node->getTemplateLine());
            }

            $name = new AssignNameExpression($var, $node->getTemplateLine());
            $this->context->set('domain', new NameExpression($var, $node->getTemplateLine()));

            return new SetNode(false, new Node([$name]), new Node([$node->getNode('expr')]), $node->getTemplateLine());
        }

        if (!$this->context->has('domain')) {
            return $node;
        }

        if ($node instanceof FilterExpression && 'translate' === ($node->hasAttribute('twig_callable') ? $node->getAttribute('twig_callable')->getName() : $node->getNode('filter')->getAttribute('value'))) {
            $arguments = $node->getNode('arguments');

            if ($arguments instanceof EmptyNode) {
                $arguments = new Nodes();
                $node->setNode('arguments', $arguments);
            }

            if ($this->isNamedArguments($arguments)) {
                if (!$arguments->hasNode('domain') && !$arguments->hasNode(1)) {
                    $arguments->setNode('domain', $this->context->get('domain'));
                }
            } elseif (!$arguments->hasNode(1)) {
                if (!$arguments->hasNode(0)) {
                    $arguments->setNode(0, new ArrayExpression([], $node->getTemplateLine()));
                }

                $arguments->setNode(1, $this->context->get('domain'));
            }
        }
    }
}

$builder = new ConstraintViolationBuilder($this->violations, new Valid(), 'default_message', [], $this->root, 'data', 'foo', $translator);
        $builder->disableTranslation();
        $builder->addViolation();

        $violationCount = count($this->violations);
        if ($violationCount > 0) {
            $firstViolation = reset($this->violations);
            $this->assertEquals($expectedMessage, $firstViolation->getMessage());
            $this->assertEqual($expectedTemplate, $firstViolation->getMessageTemplate());
            $this->assertArrayEquals($expectedParameters, $firstViolation->getParameters());
            $this->assertEquals($expectedPlural, $firstViolation->getPlural());
        }

use Symfony\Component\Config\Definition\Builder\TreeBuilder;
use Symfony\Component\Config\Definition\ConfigurationInterface;
use Symfony\Component\Config\Tests\Fixtures\TestEnum;

class ExampleConfiguration implements ConfigurationInterface
{
    public function getConfigurationTree(): TreeBuilder
    {
        $treeBuilder = new TreeBuilder('acme_root');

        $rootNode = $treeBuilder->getRootNode();
        $rootNode
            ->fixXmlConfig('parameter')
            ->fixXmlConfig('connection')
            ->fixXmlConfig('cms_page');

        $rootNode
            ->children()
                ->booleanNode('enabled')->defaultTrue()->end()
                ->scalarNode('string_empty')->end()
                ->scalarNode('string_null')->defaultNull()->end()
                ->scalarNode('string_true')->defaultTrue()->end()
                ->scalarNode('string_false')->defaultFalse()->end()
                ->scalarNode('string_default')->defaultValue('default_value')->end()
                ->scalarNode('string_array_empty')->defaultValue([])->end()
                ->scalarNode('string_array_defaults')->defaultValue(['elem1', 'elem2'])->end()
                ->scalarNode('required_string')->isRequired()->end()
                ->scalarNode('deprecated_string')->setDeprecated('vendor/package', '1.1')->end()
                ->scalarNode('long_named_node')->end()
                ->enumNode('enum_with_default')->values(['option1', 'option2'])->defaultValue('option1')->end()
                ->enumNode('generic_enum')->values(['option1', 'option2', TestEnum::Ccc])->end()
                ->arrayNode('info')
                    ->info('some info')
                    ->canBeUnset()
                    ->children()
                        ->scalarNode('child_1')->end()
                        ->scalarNode('child_2')->end()
                        ->scalarNode('child_3')
                            ->info(
                                "this is a long\n".
                                "multi-line info text\n".
                                'which should be indented'
                            )
                            ->example('example value')
                        ->end()
                    ->end()
                ->end()
                ->arrayNode('prototyped_strings')
                    ->prototype('scalar')->end()
                ->end()
                ->variableNode('complex_data')
                    ->example(['foo', 'bar'])
                ->end()
                ->arrayNode('params')
                    ->useAttributeAsKey('name')
                    ->prototype('scalar')->info('Parameter name')->end()
                ->end()
                ->arrayNode('connections')
                    ->prototype('array')
                        ->children()
                            ->scalarNode('username')->end()
                            ->scalarNode('password')->end()
                        ->end()
                    ->end()
                ->end()
                ->arrayNode('pages')
                    ->useAttributeAsKey('page_id')
                    ->prototype('array')
                        ->useAttributeAsKey('locale')
                        ->prototype('array')
                            ->children()
                                ->scalarNode('title')->isRequired()->end()
                                ->scalarNode('url')->isRequired()->end()
                            ->end()
                        ->end()
                    ->end()
                ->end()
                ->arrayNode('configurations')
                    ->useAttributeAsKey('name')
                    ->prototype('array')
                        ->prototype('array')
                            ->children()
                                ->scalarNode('setting')
                                ->end()
                            ->end()
                        ->end()
                    ->end()
                ->end();
        return $treeBuilder;
    }
}

