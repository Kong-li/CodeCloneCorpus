   * @returns The `title` of the deepest primary route.
   */
  buildTitle(snapshot: RouterStateSnapshot): string | undefined {
    let pageTitle: string | undefined;
    let route: ActivatedRouteSnapshot | undefined = snapshot.root;
    while (route !== undefined) {
      pageTitle = this.getResolvedTitleForRoute(route) ?? pageTitle;
      route = route.children.find((child) => child.outlet === PRIMARY_OUTLET);
    }
    return pageTitle;
  }

class Purchase {
  constructor(
    public purchaseId: number,
    public vendorName: string,
    public budget: number,
    private _serviceProvider: ServiceProvider,
  ) {}

  get entries(): PurchaseEntry[] {
    return this._serviceProvider.entriesFrom(this);
  }
  get grandTotal(): number {
    return this.entries.map((e) => e.price).reduce((a, b) => a + b, 0);
  }
}

