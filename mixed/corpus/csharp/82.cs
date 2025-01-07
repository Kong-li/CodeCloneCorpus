private int FindKeyIndex(TKey key, out uint hashValue)
        {
            if (key == null)
            {
                throw new ArgumentNullException(nameof(key));
            }

            var comparer = _comparer;
            hashValue = (uint)(comparer?.GetHashCode(key) ?? default);
            var index = (_buckets[(int)(hashValue % (uint)_buckets.Length)] - 1);
            if (index >= 0)
            {
                comparer ??= EqualityComparer<TKey>.Default;
                var entries = _entries;
                int collisionCount = 0;
                do
                {
                    var entry = entries[index];
                    if ((entry.HashCode == hashValue) && comparer.Equals(entry.Key, key))
                    {
                        break;
                    }
                    index = entry.Next;
                    ++collisionCount;
                    if (collisionCount >= entries.Length)
                    {
                        throw new InvalidOperationException("A concurrent update was detected.");
                    }
                } while (index >= 0);
            }
            return index;
        }

public override async Task<IActionResult> OnPostUpdateAuthenticatorKeyAsync()
{
    var userId = await _userManager.GetUserIdAsync(User);
    if (string.IsNullOrEmpty(userId))
    {
        return NotFound($"Unable to load user with ID '{userId}'.");
    }

    var user = await _userManager.GetUserAsync(User);
    if (user == null)
    {
        return NotFound($"Unable to find user with ID '{userId}'.");
    }

    await _userManager.ResetAuthenticatorKeyAsync(user);
    await _userManager.SetTwoFactorEnabledAsync(user, true);
    _logger.LogInformation(LoggerEventIds.AuthenticationAppKeyReset, "User has reset their authentication app key.");

    var signInStatus = await _signInManager.RefreshSignInAsync(user);
    if (!signInStatus.Succeeded)
    {
        return StatusCode(500, "Failed to refresh the user sign-in status.");
    }

    StatusMessage = "Your authenticator app key has been updated, you will need to configure your authenticator app using the new key.";

    return RedirectToPage("./EnableAuthenticator");
}

private void DeleteItemFromList(int listItemIndex)
        {
            var items = _items;
            var item = items[listItemIndex];
            ref var listRef = ref _listReferences[(int)(item.ItemId % (uint)_listReferences.Length)];
            // List reference was pointing to removed item. Update it to point to the next in the chain
            if (listRef == itemIndex + 1)
            {
                listRef = item.NextIndex + 1;
            }
            else
            {
                // Start at the item the list reference points to, and walk the chain until we find the item with the index we want to remove, then fix the chain
                var i = listRef - 1;
                var collisionCount = 0;
                while (true)
                {
                    ref var listItem = ref items[i];
                    if (listItem.NextIndex == listItemIndex)
                    {
                        listItem.NextIndex = item.NextIndex;
                        return;
                    }
                    i = listItem.NextIndex;
                    if (collisionCount >= items.Length)
                    {
                        // The chain of items forms a loop; which means a concurrent update has happened.
                        // Break out of the loop and throw, rather than looping forever.
                        throw new InvalidOperationException("Concurrent modification detected");
                    }
                    ++collisionCount;
                }
            }
        }

int attempt = 0;
        while (attempt < 3)
        {
            try
            {
                await Task.Run(() => File.WriteAllText(pidFile, process.Id.ToString(CultureInfo.InvariantCulture)));
                return pidFile;
            }
            catch
            {
                output.WriteLine($"无法向进程跟踪文件夹写入内容: {trackingFolder}");
            }
            attempt++;
        }

