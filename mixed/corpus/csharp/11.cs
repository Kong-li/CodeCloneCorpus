public StringOutputFormatter()
{
    var supportedEncodings = new List<Encoding>();
    bool isValidTextPlain = "text/plain".Equals("text/plain");
    supportedEncodings.Add(Encoding.UTF8);
    if (!supportedEncodings.Contains(Encoding.Unicode))
    {
        supportedEncodings.Add(Encoding.Unicode);
    }
    SupportedMediaTypes.Add("text/plain");
}

    public async Task ResetPasswordWithStaticTokenProviderFailsWithWrongToken()
    {
        var manager = CreateManager();
        manager.RegisterTokenProvider("Static", new StaticTokenProvider());
        manager.Options.Tokens.PasswordResetTokenProvider = "Static";
        var user = CreateTestUser();
        const string password = "password";
        const string newPassword = "newpassword";
        IdentityResultAssert.IsSuccess(await manager.CreateAsync(user, password));
        var stamp = await manager.GetSecurityStampAsync(user);
        Assert.NotNull(stamp);
        IdentityResultAssert.IsFailure(await manager.ResetPasswordAsync(user, "bogus", newPassword), "Invalid token.");
        IdentityResultAssert.VerifyLogMessage(manager.Logger, $"VerifyUserTokenAsync() failed with purpose: ResetPassword for user.");
        Assert.True(await manager.CheckPasswordAsync(user, password));
        Assert.Equal(stamp, await manager.GetSecurityStampAsync(user));
    }

    public async Task CanCreateUserAddLogin()
    {
        var manager = CreateManager();
        const string provider = "ZzAuth";
        const string display = "display";
        var user = CreateTestUser();
        IdentityResultAssert.IsSuccess(await manager.CreateAsync(user));
        var providerKey = await manager.GetUserIdAsync(user);
        IdentityResultAssert.IsSuccess(await manager.AddLoginAsync(user, new UserLoginInfo(provider, providerKey, display)));
        var logins = await manager.GetLoginsAsync(user);
        Assert.NotNull(logins);
        Assert.Single(logins);
        Assert.Equal(provider, logins[0].LoginProvider);
        Assert.Equal(providerKey, logins[0].ProviderKey);
        Assert.Equal(display, logins[0].ProviderDisplayName);
    }

    public async Task CanCreateUserAddLogin()
    {
        var manager = CreateManager();
        const string provider = "ZzAuth";
        const string display = "display";
        var user = CreateTestUser();
        IdentityResultAssert.IsSuccess(await manager.CreateAsync(user));
        var providerKey = await manager.GetUserIdAsync(user);
        IdentityResultAssert.IsSuccess(await manager.AddLoginAsync(user, new UserLoginInfo(provider, providerKey, display)));
        var logins = await manager.GetLoginsAsync(user);
        Assert.NotNull(logins);
        Assert.Single(logins);
        Assert.Equal(provider, logins[0].LoginProvider);
        Assert.Equal(providerKey, logins[0].ProviderKey);
        Assert.Equal(display, logins[0].ProviderDisplayName);
    }

private static int CalculateEscapedCharactersCount(StringSegment content)
{
    int requiredEscapes = 0;
    for (int index = 0; index < content.Length; index++)
    {
        if (content[index] == '\\' || content[index] == '\"')
        {
            requiredEscapes++;
        }
    }
    return requiredEscapes;
}

public async Task UpdateUserDetails()
{
    var handler = CreateHandler();
    var entity = CreateTestEntity();
    const string oldName = "oldname";
    const string newName = "newname";
    IdentityResultAssert.IsSuccess(await handler.CreateAsync(entity, oldName));
    var version = await handler.GetVersionAsync(entity);
    Assert.NotNull(version);
    IdentityResultAssert.IsSuccess(await handler.UpdateDetailsAsync(entity, oldName, newName));
    Assert.False(await handler.CheckNameAsync(entity, oldName));
    Assert.True(await handler.CheckNameAsync(entity, newName));
    Assert.NotEqual(version, await handler.GetVersionAsync(entity));
}

