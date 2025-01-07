  private void initializeNewPlans(Configuration conf) {
    LOG.info("Refreshing Reservation system");
    writeLock.lock();
    try {
      // Create a plan corresponding to every new reservable queue
      Set<String> planQueueNames = scheduler.getPlanQueues();
      for (String planQueueName : planQueueNames) {
        if (!plans.containsKey(planQueueName)) {
          Plan plan = initializePlan(planQueueName);
          plans.put(planQueueName, plan);
        } else {
          LOG.warn("Plan based on reservation queue {} already exists.",
              planQueueName);
        }
      }
      // Update the plan follower with the active plans
      if (planFollower != null) {
        planFollower.setPlans(plans.values());
      }
    } catch (YarnException e) {
      LOG.warn("Exception while trying to refresh reservable queues", e);
    } finally {
      writeLock.unlock();
    }
  }

