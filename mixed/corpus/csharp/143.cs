public void RegisterServices(IServiceCollection serviceCollection)
    {
        serviceCollection.AddMvc();

        serviceCollection.AddAuthentication(CookieScheme) // Sets the default scheme to cookies
            .AddCookie(CookieScheme, options =>
            {
                options.AccessDeniedPath = "/user/access-denied";
                options.LoginPath = "/user/login-path";
            });

        serviceCollection.AddSingleton<IConfigureOptions<CookieAuthenticationOptions>, ConfigureCustomCookie>();
    }

