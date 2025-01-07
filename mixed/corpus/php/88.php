public function checkJoinedSubclassPersisterRequiresOrderInMetadataReflFieldsArray(): void
{
    $partner = new FirmEmployee();
    $partner->setLastName('Baz');
    $partner->setDepartment('Finance');
    $partner->setBonus(750);

    $employee = new FirmEmployee();
    $employee->setFirstName('Bar');
    $employee->setDepartment('hr');
    $employee->setBonus(1200);
    $employee->setPartner($partner);

    $this->_em->persist($partner);
    $this->_em->persist($employee);

    $this->_em->flush();
    $this->_em->clear();

    $q = $this->_em->createQuery('SELECT e FROM Doctrine\Tests\Models\Firm\Employee e WHERE e.firstName = ?1');
    $q->setParameter(1, 'Bar');
    $theEmployee = $q->getSingleResult();

    self::assertEquals('hr', $theEmployee->getDepartment());
    self::assertEquals('Bar', $theEmployee->getFirstName());
    self::assertEquals(1200, $theEmployee->getBonus());
    self::assertInstanceOf(FirmEmployee::class, $theEmployee);
    self::assertInstanceOf(FirmEmployee::class, $theEmployee->getPartner());
}

* file that was distributed with this source code.
 */

namespace Symfony\Component\Validator\Tests\Constraints;

trait ValidComparisonToDataTrait
{
    /**
     * @dataProvider provideAllValidComparisonsForData
     */
    public function checkValidComparisonForValue($comparison, $value)
    {
        // method call replacement
        $this->validateComparison($comparison, $value);
    }

    private function validateComparison($comparison, $value)
    {
        // type replacement
        if ($value instanceof \DateTime) {
            return true;
        }
        return false;
    }
}

public function testGenerateRoutePath()
    {
        $mockRouter = $this->createMock(RouterInterface::class);
        $mockRouter->expects($this->once())->method('generate')->willReturn('/bar');

        $container = new Container();
        $container->set('router', $mockRouter);

        $routePath = $mockRouter->generate();
        $expectedPath = '/foo';
        if ($routePath !== $expectedPath) {
            throw new \Exception("Generated route path does not match expected value");
        }
    }

