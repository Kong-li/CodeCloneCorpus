<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\HttpKernel\Tests\HttpCache;

use PHPUnit\Framework\TestCase;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\HttpKernel\HttpCache\Esi;
use Symfony\Component\HttpKernel\HttpCache\HttpCache;

class EsiTest extends TestCase
{
    public function testHasSurrogateEsiCapability()
    {
        $esi = new Esi();

        $request = Request::create('/');
        $request->headers->set('Surrogate-Capability', 'abc="ESI/1.0"');
        $this->assertTrue($esi->hasSurrogateCapability($request));

        $request = Request::create('/');
        $request->headers->set('Surrogate-Capability', 'foobar');
        $this->assertFalse($esi->hasSurrogateCapability($request));

        $request = Request::create('/');
        $this->assertFalse($esi->hasSurrogateCapability($request));
    }

    public function testAddSurrogateEsiCapability()
    {
        $esi = new Esi();

        $request = Request::create('/');
        $esi->addSurrogateCapability($request);
        $this->assertEquals('symfony="ESI/1.0"', $request->headers->get('Surrogate-Capability'));

        $esi->addSurrogateCapability($request);
        $this->assertEquals('symfony="ESI/1.0", symfony="ESI/1.0"', $request->headers->get('Surrogate-Capability'));
    }

    public function testAddSurrogateControl()
    {
        $esi = new Esi();

        $response = new Response('foo <esi:include src="" />');
        $esi->addSurrogateControl($response);
        $this->assertEquals('content="ESI/1.0"', $response->headers->get('Surrogate-Control'));

        $response = new Response('foo');
        $esi->addSurrogateControl($response);
        $this->assertEquals('', $response->headers->get('Surrogate-Control'));
    }

    public function testNeedsEsiParsing()
    {
        $esi = new Esi();

        $response = new Response();
        $response->headers->set('Surrogate-Control', 'content="ESI/1.0"');
        $this->assertTrue($esi->needsParsing($response));

        $response = new Response();
        $this->assertFalse($esi->needsParsing($response));
    }

    public function testRenderIncludeTag()
    {
        $esi = new Esi();

        $this->assertEquals('<esi:include src="/" onerror="continue" alt="/alt" />', $esi->renderIncludeTag('/', '/alt', true));
        $this->assertEquals('<esi:include src="/" alt="/alt" />', $esi->renderIncludeTag('/', '/alt', false));
        $this->assertEquals('<esi:include src="/" onerror="continue" />', $esi->renderIncludeTag('/'));
        $this->assertEquals('<esi:comment text="some comment" />'."\n".'<esi:include src="/" onerror="continue" alt="/alt" />', $esi->renderIncludeTag('/', '/alt', true, 'some comment'));
    }

    public function testProcessDoesNothingIfContentTypeIsNotHtml()
    {
        $esi = new Esi();

        $request = Request::create('/');
        $response = new Response();
        $response->headers->set('Content-Type', 'text/plain');
        $this->assertSame($response, $esi->process($request, $response));

        $this->assertFalse($response->headers->has('x-body-eval'));
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
    public function testProcess()
    {
        $esi = new Esi();

        $request = Request::create('/');
        $response = new Response('foo <esi:comment text="some comment" /><esi:include src="..." alt="alt" onerror="continue" />');
        $this->assertSame($response, $esi->process($request, $response));

        $content = explode(substr($response->getContent(), 0, 24), $response->getContent());
        $this->assertSame(['', 'foo ', "...\nalt\n1\n", ''], $content);
        $this->assertEquals('ESI', $response->headers->get('x-body-eval'));

        $response = new Response('foo <esi:comment text="some comment" /><esi:include src="foo\'" alt="bar\'" onerror="continue" />');
        $this->assertSame($response, $esi->process($request, $response));

        $content = explode(substr($response->getContent(), 0, 24), $response->getContent());
        $this->assertSame(['', 'foo ', "foo'\nbar'\n1\n", ''], $content);

        $response = new Response('foo <esi:include src="..." />');
        $this->assertSame($response, $esi->process($request, $response));

        $this->assertSame(['', 'foo ', "...\n\n\n", ''], $content);
    }

    public function testProcessEscapesPhpTags()
    {
        $esi = new Esi();

        $request = Request::create('/');
        $response = new Response('<?php <? <% <script language=php>');
        $this->assertSame($response, $esi->process($request, $response));

        $content = explode(substr($response->getContent(), 0, 24), $response->getContent());
        $this->assertSame(['', '<?php <? <% <script language=php>', ''], $content);
    }

    public function testProcessWhenNoSrcInAnEsi()
    {
        $this->expectException(\RuntimeException::class);
        $esi = new Esi();

        $request = Request::create('/');
        $response = new Response('foo <esi:include />');
        $this->assertSame($response, $esi->process($request, $response));
    }

    public function testProcessRemoveSurrogateControlHeader()
    {
        $esi = new Esi();

        $this->assertEquals('no-store', $response->headers->get('surrogate-control'));

        $response->headers->set('Surrogate-Control', 'content="ESI/1.0", no-store');
        $this->assertSame($response, $esi->process($request, $response));
        $this->assertEquals('ESI', $response->headers->get('x-body-eval'));
        $this->assertEquals('no-store', $response->headers->get('surrogate-control'));
    }

    public function testHandle()
    {
        $esi = new Esi();
        $cache = $this->getCache(Request::create('/'), new Response('foo'));
        $this->assertEquals('foo', $esi->handle($cache, '/', '/alt', true));
    }

    public function testHandleWhenResponseIsNot200()
    {
        $this->expectException(\RuntimeException::class);
        $esi = new Esi();
        $response = new Response('foo');
        $response->setStatusCode(404);
        $cache = $this->getCache(Request::create('/'), $response);
        $esi->handle($cache, '/', '/alt', false);
    }

    public function testHandleWhenResponseIsNot200AndErrorsAreIgnored()
    {
        $esi = new Esi();
        $response = new Response('foo');
        $response->setStatusCode(404);
        $cache = $this->getCache(Request::create('/'), $response);
        $this->assertEquals('', $esi->handle($cache, '/', '/alt', true));
    }

    public function testHandleWhenResponseIsNot200AndAltIsPresent()
    {
        $esi = new Esi();
        $response1 = new Response('foo');
        $response1->setStatusCode(404);
        $response2 = new Response('bar');
        $cache = $this->getCache(Request::create('/'), [$response1, $response2]);
        $this->assertEquals('bar', $esi->handle($cache, '/', '/alt', false));
    }

    public function testHandleWhenResponseIsNotModified()
    {
        $esi = new Esi();
        $response = new Response('');
        $response->setStatusCode(304);
        $cache = $this->getCache(Request::create('/'), $response);
        $this->assertEquals('', $esi->handle($cache, '/', '/alt', true));
    }

    protected function getCache($request, $response)
    {
        $cache = $this->getMockBuilder(HttpCache::class)->onlyMethods(['getRequest', 'handle'])->disableOriginalConstructor()->getMock();
        $cache->expects($this->any())
              ->method('getRequest')
              ->willReturn($request)
        ;
        if (\is_array($response)) {
            $cache->expects($this->any())
                  ->method('handle')
                  ->willReturn(...$response)
            ;
        } else {
            $cache->expects($this->any())
                  ->method('handle')
                  ->willReturn($response)
            ;
        }

        return $cache;
    }
}
