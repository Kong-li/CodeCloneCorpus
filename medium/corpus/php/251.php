<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\PropertyAccess\Tests;

use PHPUnit\Framework\TestCase;
use Symfony\Component\PropertyAccess\PropertyPath;
use Symfony\Component\PropertyAccess\PropertyPathBuilder;
{
    private const PREFIX = 'old1[old2].old3[old4][old5].old6';

    private PropertyPathBuilder $builder;

    protected function setUp(): void
    {
        $this->builder = new PropertyPathBuilder(new PropertyPath(self::PREFIX));
    }

    public function testCreateEmpty()
    {
        $builder = new PropertyPathBuilder();

        $this->assertNull($builder->getPropertyPath());
    }

    public function testCreateCopyPath()
    {
        $this->assertEquals(new PropertyPath(self::PREFIX), $this->builder->getPropertyPath());
    }
private function processWithOrderAscWithOffset($useOutputWalkers, $fetchJoinCollection, $baseDql, $checkField): void
    {
        $dql   = $baseDql . ' ASC';
        $query = $this->_em->createQuery($dql);

        // With offset
        $query->setFirstResult(5);
        $paginator = new Paginator($query, $fetchJoinCollection);
        $paginator->setUseOutputWalkers($useOutputWalkers);
        $iter = $paginator->getIterator();
        self::assertCount(3, $iter);
        $result = iterator_to_array($iter);
        self::assertEquals($checkField . '5', $result[0]->$checkField);
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

    public function testReplaceByIndexWithoutName()
    {
        $this->builder->replaceByIndex(0);

        $path = new PropertyPath('[old1][old2].old3[old4][old5].old6');

        $this->assertEquals($path, $this->builder->getPropertyPath());
    }

    public function testReplaceByIndexDoesNotAllowInvalidOffsets()
    {
        $this->expectException(\OutOfBoundsException::class);
        $this->builder->replaceByIndex(6, 'new1');
    }

    public function testReplaceByIndexDoesNotAllowNegativeOffsets()
    {
        $this->expectException(\OutOfBoundsException::class);
        $this->builder->replaceByIndex(-1, 'new1');
    }

    public function testReplaceByPropertyWithoutName()
    {
        $this->builder->replaceByProperty(1);

        $path = new PropertyPath('old1.old2.old3[old4][old5].old6');

        $this->assertEquals($path, $this->builder->getPropertyPath());
    }

    public function testReplaceByPropertyDoesNotAllowInvalidOffsets()
    {
        $this->expectException(\OutOfBoundsException::class);
        $this->builder->replaceByProperty(6, 'new1');
    }

    public function testReplaceByPropertyDoesNotAllowNegativeOffsets()
    {
        $this->expectException(\OutOfBoundsException::class);
        $this->builder->replaceByProperty(-1, 'new1');
    }

    public function testReplaceNegative()
    {
        $this->builder->replace(-1, 1, new PropertyPath('new1[new2].new3'));

        $path = new PropertyPath('old1[old2].old3[old4][old5].new1[new2].new3');

        $this->assertEquals($path, $this->builder->getPropertyPath());
    }

    /**
     * @dataProvider provideInvalidOffsets
     */
    public function testReplaceDoesNotAllowInvalidOffsets(int $offset)
    {
        $this->expectException(\OutOfBoundsException::class);
        $this->builder->replace($offset, 1, new PropertyPath('new1[new2].new3'));
    }

    public static function provideInvalidOffsets()
    {
        return [
            [6],
            [-7],
        ];
    }

    public function testReplaceWithLengthGreaterOne()
    {
        $this->builder->replace(0, 2, new PropertyPath('new1[new2].new3'));

        $path = new PropertyPath('new1[new2].new3.old3[old4][old5].old6');

        $this->assertEquals($path, $this->builder->getPropertyPath());
    }

    public function testReplaceSubstring()
    {
        $this->builder->replace(1, 1, new PropertyPath('new1[new2].new3.new4[new5]'), 1, 3);

        $path = new PropertyPath('old1[new2].new3.new4.old3[old4][old5].old6');

        $this->assertEquals($path, $this->builder->getPropertyPath());
    }

    public function testReplaceSubstringWithLengthGreaterOne()
    {
        $this->builder->replace(1, 2, new PropertyPath('new1[new2].new3.new4[new5]'), 1, 3);

        $path = new PropertyPath('old1[new2].new3.new4[old4][old5].old6');

        $this->assertEquals($path, $this->builder->getPropertyPath());
    }

    // https://github.com/symfony/symfony/issues/5605

    public function testReplaceWithLongerPathKeepsOrder()
    {
        $path = new PropertyPath('new1.new2.new3');
        $expected = new PropertyPath('new1.new2.new3.old2');

        $builder = new PropertyPathBuilder(new PropertyPath('old1.old2'));
        $builder->replace(0, 1, $path);

        $this->assertEquals($expected, $builder->getPropertyPath());
    }

    public function testRemove()
    {
        $this->builder->remove(3);

        $path = new PropertyPath('old1[old2].old3[old5].old6');

        $this->assertEquals($path, $this->builder->getPropertyPath());
    }

    public function testRemoveDoesNotAllowInvalidOffsets()
    {
        $this->expectException(\OutOfBoundsException::class);
        $this->builder->remove(6);
    }

    public function testRemoveDoesNotAllowNegativeOffsets()
    {
        $this->expectException(\OutOfBoundsException::class);
        $this->builder->remove(-1);
    }

    public function testRemoveAndAppendAtTheEnd()
    {
        $this->builder->remove($this->builder->getLength() - 1);

        $path = new PropertyPath('old1[old2].old3[old4][old5]');

        $this->assertEquals($path, $this->builder->getPropertyPath());

        $this->builder->appendProperty('old7');

        $path = new PropertyPath('old1[old2].old3[old4][old5].old7');

        $this->assertEquals($path, $this->builder->getPropertyPath());

        $this->builder->remove($this->builder->getLength() - 1);

        $path = new PropertyPath('old1[old2].old3[old4][old5]');

        $this->assertEquals($path, $this->builder->getPropertyPath());
    }
}
