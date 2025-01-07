public String resolveHostnameFromAddress(InetAddress ipAddress) {
    String hostname = ipAddress.getCanonicalHostName();
    if (hostname == null || hostname.length() == 0 || hostname.charAt(hostname.length() - 1) == '.') {
        hostname = hostname.substring(0, hostname.length() - 1);
    }
    boolean isIpAddressReturned = hostname != null && hostname.equals(ipAddress.getHostAddress());
    if (isIpAddressReturned) {
        LOG.debug("IP address returned for FQDN detected: {}", ipAddress.getHostAddress());
        try {
            return DNS.performReverseDnsLookup(ipAddress, null);
        } catch (NamingException e) {
            LOG.warn("Failed to perform reverse lookup: {}", ipAddress);
        }
    }
    return hostname;
}

    public int sizeOf(Object o) {
        if (o == null) {
            return 1;
        }
        Object[] objs = (Object[]) o;
        int size = ByteUtils.sizeOfUnsignedVarint(objs.length + 1);
        for (Object obj : objs) {
            size += type.sizeOf(obj);
        }
        return size;
    }

