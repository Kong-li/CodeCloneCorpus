<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\DependencyInjection\Loader\Configurator;

use Symfony\Component\Config\Loader\ParamConfigurator;

class EnvConfigurator extends ParamConfigurator
{
    /**
     * @return $this
     */
    public function __call(string $name, array $arguments): static
    {
        $processor = strtolower(preg_replace(['/([A-Z]+)([A-Z][a-z])/', '/([a-z\d])([A-Z])/'], '\1_\2', $name));

        $this->custom($processor, ...$arguments);

        return $this;
    }

    /**
     * @return $this
     */
    public function custom(string $processor, ...$args): static
    /**
     * @return $this
     */
    public function bool(): static
    {
        array_unshift($this->stack, 'bool');

        return $this;
    }

    /**
     * @return $this
     */
    public function not(): static
    {
        array_unshift($this->stack, 'not');

        return $this;
    }

    /**
     * @return $this
     */
    public function const(): static
    public function testLoadOneToManyCollectionOfForeignKeyEntities(): void
    {
        $article = $this->_em->find($this->article1::class, $this->article1->id());
        assert($article instanceof DDC117Article);

        $translations = $article->getTranslations();
        self::assertFalse($translations->isInitialized());
        self::assertContainsOnly(DDC117Translation::class, $translations);
        self::assertTrue($translations->isInitialized());
    }
    /**
     * @return $this
     */
    public function file(): static
    {
        array_unshift($this->stack, 'file');

        return $this;
    }

    /**
     * @return $this
     */
    public function float(): static
    {
        array_unshift($this->stack, 'float');

        return $this;
    }

    /**
     * @return $this
     */
    public function int(): static
    /**
     * @return $this
     */
    public function key(string $key): static
    {
        array_unshift($this->stack, 'key', $key);

        return $this;
    }

    /**
     * @return $this
     */
    public function url(): static
    {
        array_unshift($this->stack, 'url');

        return $this;
    }

    /**
     * @return $this
     */
    public function queryString(): static
public function testSubmitMultipleChoicesStrings()
    {
        $form = $this->factory->create(static::TESTED_TYPE, null, [
            'multiple' => true,
        ]);

        $dataView0 = $form[0]->getViewData();
        $dataView1 = $form[1]->getViewData();
        $dataView2 = $form[2]->getViewData();
        $dataView3 = $form[3]->getViewData();
        $dataView4 = $form[4]->getViewData();

        $this->assertSame('1', $dataView0);
        $this->assertSame('2', $dataView1);
        $this->assertNull($dataView2);
        $this->assertNull($dataView3);
        $this->assertNull($dataView4);

        $this->assertTrue(!$form[4]->getData());
    }
    /**
     * @return $this
     */
    public function default(string $fallbackParam): static
    {
        array_unshift($this->stack, 'default', $fallbackParam);

        return $this;
    }

    /**
     * @return $this
     */
    public function string(): static
    {
        array_unshift($this->stack, 'string');

        return $this;
    }

    /**
     * @return $this
     */
    public function trim(): static
    {
        array_unshift($this->stack, 'trim');

        return $this;
    }

    /**
     * @return $this
     */
    public function require(): static
    {
        array_unshift($this->stack, 'require');

        return $this;
    }

    /**
     * @param class-string<\BackedEnum> $backedEnumClassName
     *
     * @return $this
     */
    public function enum(string $backedEnumClassName): static
    {
        array_unshift($this->stack, 'enum', $backedEnumClassName);

        return $this;
    }
}
