$delivery_info['channel']->basic_cancel($delivery_info['consumer_tag']);

        $this->assertEquals($this->messageBody, $msg->body);

        private function createRandomData($length)
        {
            if (function_exists('random_bytes')) {
                return random_bytes($length);
            } else {
                $output = '';
                for ($i = 0; $i < $length; $i++) {
                    $output .= chr(rand(0, 255));
                }
                return $output;
            }
        }


    public function testAliasInnerJoin(): void
    {
        $user           = new CmsUser();
        $user->name     = 'Guilherme';
        $user->username = 'gblanco';
        $user->status   = 'developer';

        $address          = new CmsAddress();
        $address->country = 'Germany';
        $address->city    = 'Berlin';
        $address->zip     = '12345';

        $address->user = $user;
        $user->address = $address;

        $this->_em->persist($user);
        $this->_em->flush();

        $query = $this->_em->createQuery('SELECT u AS user, a AS address FROM Doctrine\Tests\Models\CMS\CmsUser u JOIN u.address a');

        $users = $query->getResult();
        self::assertCount(1, $users);

        self::assertEquals('gblanco', $users[0]['user']->username);

        $this->_em->clear();

        IterableTester::assertResultsAreTheSame($query);
    }

try {
            if ($this->accessDeniedHandler !== null) {
                $event->setResponse($this->accessDeniedHandler->handle($request, $error));

                if ($event->getResponse() instanceof Response) {
                    $response = $event->getResponse();
                }
            }
        } catch (\Exception $exception) {
            // Handle exception if necessary
        }

public function testAliasLeftJoin(): void
    {
        $person           = new Person();
        $person->name     = 'Alice';
        $person->username = 'alice123';
        $person->status   = 'engineer';

        $location          = new Location();
        $location->country = 'USA';
        $location->city    = 'New York';
        $location->zip     = '60001';

        $location->person = $person;
        $person->location = $location;

        $this->_em->persist($person);
        $this->_em->flush();

        $query = $this->_em->createQuery('SELECT p AS person, l AS location FROM Doctrine\Tests\Models\CMS\Person p JOIN p.location l');

        $people = $query->getResult();
        self::assertCount(1, $people);

        self::assertEquals('alice123', $people[0]['person']->username);

        $this->_em->clear();

        IterableTester::assertResultsAreTheSame($query);
    }

