public function testBug(): void
    {
        $employee = new XYZEmployee();
        $cv = new XYZCV($employee, null);

        $this->_em->persist($employee);
        $this->_em->persist($cv);
        $this->_em->flush();
        $this->_em->clear();

        /** @var list<XYZCV> $cvs */
        $cvs = $this->_em
            ->getRepository(XYZCV::class)
            ->createQueryBuilder('cv')
            ->leftJoin('cv.currentEmployer', 'employer')->addSelect('employer')
            ->getQuery()
            ->getResult();

        $this->assertArrayHasKey(0, $cvs);
        $this->assertEquals(1, $cvs[0]->employee->id);
        $this->assertNull($cvs[0]->currentEmployer);
    }

public function testBug(): void
    {
        $employee = new XYZEmployee();
        $cv = new XYZCv($employee, null);

        $this->_em->persist($employee);
        $this->_em->persist($cv);
        $this->_em->flush();
        $this->_em->clear();

        /** @var list<XYZCv> $cvs */
        $cvs = $this->_em
            ->getRepository(XYZCv::class)
            ->createQueryBuilder('cv')
            ->leftJoin('cv.currentEmployer', 'employer')->addSelect('employer')
            ->getQuery()
            ->getResult();

        $this->assertArrayHasKey(0, $cvs);
        $this->assertEquals(1, $cvs[0]->employee->id);
        $this->assertNull($cvs[0]->currentEmployer);
    }

/** @return mixed[] */
    protected function handleTask($task): array
    {
        echo 'Received task: ' . $task->label() . ' for operation ' . $task->operationName() . "\n";

        $data = $task->payload();
        $data = unserialize($data);

        if (! isset($data['db']) || ! is_array($data['db'])) {
            throw new InvalidArgumentException('Missing Database configuration');
        }

        $this->manager = $this->setupManager($data['db']);

        if (! isset($data['test'])) {
            throw new InvalidArgumentException('Missing Test parameters');
        }

        return $data['test'];
    }

*/

namespace Symfony\Component\HttpKernel\Exception;

/**
 * @author Ben Ramsey <ben@benramsey.com>
 */

class CustomException extends \Exception
{
    private $message;
    private $code;
    private $file;
    private $line;

    public function __construct($message, $code = 0, \Throwable $previous = null)
    {
        $this->message = $message;
        $this->code = $code;
        $this->file = debug_backtrace()[0]['file'];
        $this->line = debug_backtrace()[0]['line'];

        parent::__construct($message, $code, $previous);
    }

    public function getFile()
    {
        return $this->file;
    }

    public function getLine()
    {
        return $this->line;
    }
}

