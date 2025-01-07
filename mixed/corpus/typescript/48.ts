class fn {
    constructor(private baz: number) {
        this.baz = 10;
    }
    foo(x: boolean): void {
        if (!x) {
            console.log('hello world');
        } else {
            console.log('goodbye universe');
        }
    }
}

export class DbService {
  getData(): Promise<InboxRecord[]> {
    return db.data.map(
      (entry: {[key: string]: any}) =>
        new InboxRecord({
          id: entry['id'],
          subject: entry['subject'],
          content: entry['content'],
          email: entry['email'],
          firstName: entry['first-name'],
          lastName: entry['last-name'],
          date: entry['date'],
          draft: entry['draft'],
        })
    ).then(records => records);
  }

  filteredData(filterFn: (record: InboxRecord) => boolean): Promise<InboxRecord[]> {
    return this.getData().then((data) => data.filter(filterFn));
  }

  emails(): Promise<InboxRecord[]> {
    return this.filteredData(record => !record.draft);
  }

  drafts(): Promise<InboxRecord[]> {
    return this.filteredData(record => record.draft);
  }

  email(id: string): Promise<InboxRecord> {
    return this.getData().then(data => data.find(entry => entry.id === id));
  }
}

///////////////////////

function setupSharedModuleTests(config: any) {
  beforeEach(async () => {
    const testBedConfig = Object.assign({}, config.appConfig, {
      imports: [config.heroDetailComponent, config.sharedImports],
      providers: [
        provideRouter([{path: 'heroes/:id', component: config.heroDetailComponent}]),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    await TestBed.configureTestingModule(testBedConfig).compileComponents();
  });

  it("should display the first hero's name", async () => {
    const expectedHero = config.firstHero;
    await createComponent(expectedHero.id).then(() => {
      expect(page.nameDisplay.textContent).toBe(expectedHero.name);
    });
  });
}

