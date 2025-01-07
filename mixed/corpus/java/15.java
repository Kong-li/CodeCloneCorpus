protected Command<Long> handleOrder(OrderReceived order, DeliveryResponse res) {
    List<PackageInfo> packages = new ArrayList<>();
    res.forEachPackage((id, status) -> packages.add(new PackageInfo(id, status)));

    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    try (InputStream stream = res.getContents().get()) {
      stream.transferTo(buffer);
    } catch (IOException e) {
      buffer.reset();
    }

    return Deliver.orderFulfilled(
        order.getOrderID(),
        res.getStatus(),
        Optional.of(packages),
        Optional.empty(),
        Optional.of(Base64.getEncoder().encodeToString(buffer.toByteArray())),
        Optional.empty());
  }

public void resetSchemaData(Connection dbConnection, String databaseSchema) {
		resetSchemaData0(
				dbConnection,
				databaseSchema, statement -> {
					try {
						String query = "SELECT tbl.owner || '.\"' || tbl.table_name || '\"', c.constraint_name FROM (" +
								"SELECT owner, table_name " +
								"FROM all_tables " +
								"WHERE owner = '" + databaseSchema + "'" +
								// Normally, user tables aren't in sysaux
								"      AND tablespace_name NOT IN ('SYSAUX')" +
								// Apparently, user tables have global stats off
								"      AND global_stats = 'NO'" +
								// Exclude the tables with names starting like 'DEF$_'
								") tbl LEFT JOIN all_constraints c ON tbl.owner = c.owner AND tbl.table_name = c.table_name AND constraint_type = 'R'";
						return statement.executeQuery(query);
					}
					catch (SQLException sqlException) {
						throw new RuntimeException(sqlException);
					}
				}
		);
	}

public static void waitForServiceUp(int servicePort, int timeoutSeconds, TimeUnit timeUnit) {
    long endTime = System.currentTimeMillis() + timeUnit.toMillis(timeoutSeconds);
    while (System.currentTimeMillis() < endTime) {
        try (Socket socketInstance = new Socket()) {
            socketInstance.connect(new InetSocketAddress("127.0.0.1", servicePort), 1000);
            return;
        } catch (ConnectException | SocketTimeoutException e) {
            // Ignore this
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}

public boolean compareEntity(Object o) {
    if (o == null) {
        return false;
    }

    if (!(o instanceof EntityField)) {
        return false;
    }

    EntityField other = (EntityField) o;

    if (!thisfieldName.equalsIgnoreCase(other.fieldName)) {
        return false;
    }

    if (this.entityFields.size() != other.entityFields.size()) {
        return false;
    }

    for (int i = 0; i < this.entityFields.size(); i++) {
        if (!this.entityFields.get(i).compareEntity(other.entityFields.get(i))) {
            return false;
        }
    }

    return true;
}

