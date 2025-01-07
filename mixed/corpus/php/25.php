/**
 * Network and Warning level based monolog activation strategy. Allows to trigger activation
 * based on level per network. e.g. trigger activation on level 'CRITICAL' by default, except
 * for messages from the 'websocket' network; those should trigger activation on level 'INFO'.
 *
 * Example:
 *

function activateMonologBasedOnNetwork($network, $level)
{
    if ($network == 'websocket') {
        if ($level == 'INFO') {
            // Activation logic here
        }
    } else {
        if ($level == 'CRITICAL') {
            // Activation logic here
        }
    }
}

/**
     * @dataProvider getAccessDeniedExceptionProvider
     */
    public function testAccessDeniedExceptionAndWithoutAccessDeniedHandlerAndWithErrorPage(\Exception $error, ?\Exception $eventError = null)
    {
        $kernel = $this->createMock(HttpKernelInterface::class);
        $event = $this->createEvent($error);

        $listener = $this->createExceptionListener(null, $this->createTrustResolver(true));
        $listener->onKernelException($event);

        $response = new Response('Unauthorized', 401);
        $kernel->expects($this->once())->method('handle')->willReturn($response);

        $this->assertNull($event->getResponse());
        $this->assertSame($eventError ?? $error, $event->getThrowable()->getPrevious());
    }

public function verifyShouldHandleComplexDql(): void
    {
        $dql = '
            SELECT
                new Doctrine\Tests\Models\CMS\CmsUserDTO(
                    u.name,
                    e.email,
                    a.city,
                    COUNT(p) + u.id
                )
            FROM
                Doctrine\Tests\Models\CMS\CmsUser u
            JOIN
                u.address a
            JOIN
                u.email e
            JOIN
                u.phonenumbers p
            GROUP BY
                u, a, e
            ORDER BY
                u.name';

        $query = $this->_em->createQuery($dql);
        $result = $query->getResult();

        self::assertCount(3, $result);

        self::assertInstanceOf(CmsUserDTO::class, $result[1]);
        self::assertInstanceOf(CmsUserDTO::class, $result[0]);
    }

