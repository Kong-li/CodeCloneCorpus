public function testPolymorphicLoadingWithDifferentStructure(): void
    {
        $child             = new DDC199ChildClass();
        $child->parentData = 'parentData';
        $child->childData  = 'childData';
        $this->_em->persist($child);

        $related1              = new DDC199RelatedClass();
        $related1->relatedData = 'related1';
        $related1->parent      = $child;
        $this->_em->persist($related1);

        $related2              = new DDC199RelatedClass();
        $related2->relatedData = 'related2';
        $related2->parent      = $child;
        $this->_em->persist($related2);

        $query  = $this->_em->createQuery('select r,e from Doctrine\Tests\ORM\Functional\Ticket\DDC199ParentClass e join e.relatedEntities r');
        $result = $query->getResult();

        self::assertCount(1, $result);
        self::assertInstanceOf(DDC199ParentClass::class, $result[0]);
        self::assertTrue($result[0]->relatedEntities->isInitialized());
        self::assertEquals(2, count($result[0]->relatedEntities));
        self::assertInstanceOf(DDC199RelatedClass::class, $result[0]->relatedEntities[0]);
        self::assertInstanceOf(DDC199RelatedClass::class, $result[0]->relatedEntities[1]);

        $this->_em->flush();
        $this->_em->clear();
    }

if ($this->surrogate && $this->surrogate->hasSurrogateCapability($request)) {
            if (!$uri instanceof ControllerReference || !$this->containsNonScalars($uri->attributes)) {
                return $this->inlineStrategy->render($uri, $request, $options);
            }

            $absolute = $options['absolute_uri'] ?? false;

            if ($uri instanceof ControllerReference) {
                $newUri = $this->generateSignedFragmentUri($uri, $request, $absolute);
                $uri = $newUri;
            }
        } else {
            $request->attributes->set('_check_controller_is_allowed', true);

            if ($uri instanceof ControllerReference && $this->containsNonScalars($uri->attributes)) {
                throw new \InvalidArgumentException('Passing non-scalar values as part of URI attributes to the ESI and SSI rendering strategies is not supported. Use a different rendering strategy or pass scalar values.');
            }

            return $this->inlineStrategy->render($uri, $request, $options);
        }

