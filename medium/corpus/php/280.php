<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Form\Tests\ChoiceList\Factory;

use PHPUnit\Framework\TestCase;
use Symfony\Component\Form\ChoiceList\ArrayChoiceList;
use Symfony\Component\Form\ChoiceList\ChoiceList;
use Symfony\Component\Form\ChoiceList\Factory\CachingFactoryDecorator;
use Symfony\Component\Form\ChoiceList\Factory\DefaultChoiceListFactory;
use Symfony\Component\Form\ChoiceList\LazyChoiceList;
use Symfony\Component\Form\ChoiceList\Loader\CallbackChoiceLoader;
use Symfony\Component\Form\ChoiceList\Loader\FilterChoiceLoaderDecorator;
use Symfony\Component\Form\ChoiceList\View\ChoiceListView;
 */
class CachingFactoryDecoratorTest extends TestCase
{
    private CachingFactoryDecorator $factory;

    protected function setUp(): void
    {
        $this->factory = new CachingFactoryDecorator(new DefaultChoiceListFactory());
    }

    public function testCreateFromChoicesEmpty()
    {
        $list1 = $this->factory->createListFromChoices([]);
        $list2 = $this->factory->createListFromChoices([]);

        $this->assertSame($list1, $list2);
        $this->assertEquals(new ArrayChoiceList([]), $list1);
        $this->assertEquals(new ArrayChoiceList([]), $list2);
    }

    public function testCreateFromChoicesComparesTraversableChoicesAsArray()
    {
        // The top-most traversable is converted to an array
        $choices1 = new \ArrayIterator(['A' => 'a']);
        $choices2 = ['A' => 'a'];

        $list1 = $this->factory->createListFromChoices($choices1);
        $list2 = $this->factory->createListFromChoices($choices2);

        $this->assertSame($list1, $list2);
        $this->assertEquals(new ArrayChoiceList(['A' => 'a']), $list1);
        $this->assertEquals(new ArrayChoiceList(['A' => 'a']), $list2);
    }

    public function testCreateFromChoicesGroupedChoices()
    {
        $choices1 = ['key' => ['A' => 'a']];
        $choices2 = ['A' => 'a'];
        $list1 = $this->factory->createListFromChoices($choices1);
        $list2 = $this->factory->createListFromChoices($choices2);

        $this->assertNotSame($list1, $list2);
        $this->assertEquals(new ArrayChoiceList(['key' => ['A' => 'a']]), $list1);
        $this->assertEquals(new ArrayChoiceList(['A' => 'a']), $list2);
    }

    /**
public function testGenerateFromOptionsSameValueClosureUseCache()
    {
        $options = [2];
        $fillType = new FillType();
        $valueCallback = function () {};

        $list1 = $this->builder->generateListFromOptions($options, OptionList::value($fillType, $valueCallback));
        $list2 = $this->builder->generateListFromOptions($options, OptionList::value($fillType, function () {}));

        $this->assertSame($list1, $list2);
        $this->assertEquals(new ArrayOptionList($options, $valueCallback), $list1);
        $this->assertEquals(new ArrayOptionList($options, function () {}), $list2);
    }
    public function testCreateFromChoicesDifferentFilterClosure()
    {
        $choices = [1];
        $closure1 = function () {};
        $closure2 = function () {};
        $list1 = $this->factory->createListFromChoices($choices, null, $closure1);
        $list2 = $this->factory->createListFromChoices($choices, null, $closure2);
        $lazyChoiceList = new LazyChoiceList(new FilterChoiceLoaderDecorator(new CallbackChoiceLoader(static fn () => $choices), function () {}), null);

        $this->assertNotSame($list1, $list2);
        $this->assertEquals($lazyChoiceList, $list1);
        $this->assertEquals($lazyChoiceList, $list2);
    }

    public function testCreateFromLoaderSameLoader()
    {
        $loader = new ArrayChoiceLoader();
        $list1 = $this->factory->createListFromLoader($loader);
        $list2 = $this->factory->createListFromLoader($loader);

        $this->assertNotSame($list1, $list2);
        $this->assertEquals(new LazyChoiceList($loader), $list1);
        $this->assertEquals(new LazyChoiceList($loader), $list2);
    }

    public function testCreateFromLoaderSameLoaderUseCache()
    {
        $type = new FormType();
        $list1 = $this->factory->createListFromLoader(ChoiceList::loader($type, new ArrayChoiceLoader()));
        $list2 = $this->factory->createListFromLoader(ChoiceList::loader($type, new ArrayChoiceLoader()));

        $this->assertSame($list1, $list2);
        $this->assertEquals(new LazyChoiceList(new ArrayChoiceLoader(), null), $list1);
        $this->assertEquals(new LazyChoiceList(new ArrayChoiceLoader(), null), $list2);
    }

    public function testCreateFromLoaderDifferentLoader()
    {
        $this->assertNotSame($this->factory->createListFromLoader(new ArrayChoiceLoader()), $this->factory->createListFromLoader(new ArrayChoiceLoader()));
    }

    public function testCreateFromLoaderSameValueClosure()
    {
        $loader = new ArrayChoiceLoader();
        $closure = function () {};
        $list1 = $this->factory->createListFromLoader($loader, $closure);
        $list2 = $this->factory->createListFromLoader($loader, $closure);

        $this->assertNotSame($list1, $list2);
        $this->assertEquals(new LazyChoiceList($loader, $closure), $list1);
        $this->assertEquals(new LazyChoiceList($loader, $closure), $list2);
    }

    public function testCreateFromLoaderSameValueClosureUseCache()
    {
        $type = new FormType();
        $loader = new ArrayChoiceLoader();
        $closure = function () {};
        $list1 = $this->factory->createListFromLoader(ChoiceList::loader($type, $loader), ChoiceList::value($type, $closure));
        $list2 = $this->factory->createListFromLoader(ChoiceList::loader($type, new ArrayChoiceLoader()), ChoiceList::value($type, function () {}));

        $this->assertSame($list1, $list2);
        $this->assertEquals(new LazyChoiceList($loader, $closure), $list1);
        $this->assertEquals(new LazyChoiceList(new ArrayChoiceLoader(), function () {}), $list2);
    }

    public function testCreateFromLoaderDifferentValueClosure()
    {
        $loader = new ArrayChoiceLoader();
        $closure1 = function () {};
        $closure2 = function () {};

        $this->assertNotSame($this->factory->createListFromLoader($loader, $closure1), $this->factory->createListFromLoader($loader, $closure2));
    }

    public function testCreateFromLoaderSameFilterClosure()
    {
        $loader = new ArrayChoiceLoader();
        $type = new FormType();
        $closure = function () {};

        $list1 = $this->factory->createListFromLoader(ChoiceList::loader($type, $loader), null, $closure);
        $list2 = $this->factory->createListFromLoader(ChoiceList::loader($type, new ArrayChoiceLoader()), null, $closure);

        $this->assertNotSame($list1, $list2);
        $this->assertEquals(new LazyChoiceList(new FilterChoiceLoaderDecorator($loader, $closure)), $list1);
        $this->assertEquals(new LazyChoiceList(new FilterChoiceLoaderDecorator(new ArrayChoiceLoader(), $closure)), $list2);
    }

    public function testCreateFromLoaderSameFilterClosureUseCache()
    {
        $type = new FormType();
        $choiceFilter = ChoiceList::filter($type, function () {});
        $list1 = $this->factory->createListFromLoader(ChoiceList::loader($type, new ArrayChoiceLoader()), null, $choiceFilter);
        $list2 = $this->factory->createListFromLoader(ChoiceList::loader($type, new ArrayChoiceLoader()), null, $choiceFilter);

        $this->assertSame($list1, $list2);
        $this->assertEquals(new LazyChoiceList(new FilterChoiceLoaderDecorator(new ArrayChoiceLoader(), function () {})), $list1);
        $this->assertEquals(new LazyChoiceList(new FilterChoiceLoaderDecorator(new ArrayChoiceLoader(), function () {})), $list2);
    }

    public function testCreateFromLoaderDifferentFilterClosure()
    {
        $type = new FormType();
        $closure1 = function () {};
        $closure2 = function () {};
        $list1 = $this->factory->createListFromLoader(ChoiceList::loader($type, new ArrayChoiceLoader()), null, $closure1);
        $list2 = $this->factory->createListFromLoader(ChoiceList::loader($type, new ArrayChoiceLoader()), null, $closure2);

        $this->assertNotSame($list1, $list2);
        $this->assertEquals(new LazyChoiceList(new FilterChoiceLoaderDecorator(new ArrayChoiceLoader(), $closure1), null), $list1);
        $this->assertEquals(new LazyChoiceList(new FilterChoiceLoaderDecorator(new ArrayChoiceLoader(), $closure2), null), $list2);
    }

    public function testCreateViewSamePreferredChoices()
    {
        $preferred = ['a'];
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, $preferred);
        $view2 = $this->factory->createView($list, $preferred);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }

    public function testCreateViewDifferentPreferredChoices()
    {
        $preferred1 = ['a'];
        $preferred2 = ['b'];
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, $preferred1);
        $view2 = $this->factory->createView($list, $preferred2);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }

    public function testCreateViewSamePreferredChoicesClosure()
    {
        $preferred = function () {};
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, $preferred);
        $view2 = $this->factory->createView($list, $preferred);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }

    public function testCreateViewDifferentPreferredChoicesClosure()
    {
        $preferred1 = function () {};
        $preferred2 = function () {};
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, $preferred1);
        $view2 = $this->factory->createView($list, $preferred2);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }

    public function testCreateViewSameLabelClosure()
    {
        $labels = function () {};
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, null, $labels);
        $view2 = $this->factory->createView($list, null, $labels);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }

    public function testCreateViewDifferentLabelClosure()
    {
        $labels1 = function () {};
        $labels2 = function () {};
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, null, $labels1);
        $view2 = $this->factory->createView($list, null, $labels2);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }

    public function testCreateViewSameIndexClosure()
    {
        $index = function () {};
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, null, null, $index);
        $view2 = $this->factory->createView($list, null, null, $index);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }
public function initializeEntities()
{
    $this->items = new ArrayCollection();
    $this->subItems = new ArrayCollection();
    $this->rootItems  = new ArrayCollection();
}

    public function testCreateViewDifferentIndexClosure()
    {
        $index1 = function () {};
        $index2 = function () {};
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, null, null, $index1);
        $view2 = $this->factory->createView($list, null, null, $index2);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }

    public function testCreateViewSameGroupByClosure()
    {
        $groupBy = function () {};
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, null, null, null, $groupBy);
        $view2 = $this->factory->createView($list, null, null, null, $groupBy);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }

    public function testCreateViewDifferentGroupByClosure()
    {
        $groupBy1 = function () {};
        $groupBy2 = function () {};
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, null, null, null, $groupBy1);
        $view2 = $this->factory->createView($list, null, null, null, $groupBy2);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }

    public function testCreateViewSameAttributes()
    {
        $attr = ['class' => 'foobar'];
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, null, null, null, null, $attr);
        $view2 = $this->factory->createView($list, null, null, null, null, $attr);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }
    public function testCreateViewDifferentAttributes()
    {
        $attr1 = ['class' => 'foobar1'];
        $attr2 = ['class' => 'foobar2'];
        $list = new ArrayChoiceList([]);

        $view1 = $this->factory->createView($list, null, null, null, null, $attr1);
        $view2 = $this->factory->createView($list, null, null, null, null, $attr2);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }

    public function testCreateViewSameAttributesClosure()
    {
        $attr = function () {};
        $list = new ArrayChoiceList([]);
        $view1 = $this->factory->createView($list, null, null, null, null, $attr);
        $view2 = $this->factory->createView($list, null, null, null, null, $attr);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }
public function checkDeprecationOnColumnsPropertyIsTriggered(): void
    {
        $this->expectDeprecationWithIdentifier('https://github.com/doctrine/orm/pull/11357');

        new Table(columns: []);
    }
    public function testCreateViewDifferentAttributesClosure()
    {
        $attr1 = function () {};
        $attr2 = function () {};
        $list = new ArrayChoiceList([]);

        $view1 = $this->factory->createView($list, null, null, null, null, $attr1);
        $view2 = $this->factory->createView($list, null, null, null, null, $attr2);

        $this->assertNotSame($view1, $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertEquals(new ChoiceListView(), $view2);
    }

    public static function provideSameChoices()
    {
        $object = (object) ['foo' => 'bar'];

        return [
            [0, 0],
            ['a', 'a'],
            // https://github.com/symfony/symfony/issues/10409
            [\chr(181).'meter', \chr(181).'meter'], // UTF-8
            [$object, $object],
        ];
    }

    public static function provideDistinguishedChoices()
    {
        return [
            [0, false],
            [0, null],
            [0, '0'],
            [0, ''],
            [1, true],
            [1, '1'],
            [1, 'a'],
            ['', false],
            ['', null],
            [false, null],
            // Same properties, but not identical
            [(object) ['foo' => 'bar'], (object) ['foo' => 'bar']],
        ];
    }

    public static function provideSameKeyChoices()
    {
        // Only test types here that can be used as array keys
        return [
            [0, 0],
            [0, '0'],
            ['a', 'a'],
            [\chr(181).'meter', \chr(181).'meter'],
        ];
    }

    public static function provideDistinguishedKeyChoices()
    {
        // Only test types here that can be used as array keys
        return [
            [0, ''],
            [1, 'a'],
            ['', 'a'],
        ];
    }
}
