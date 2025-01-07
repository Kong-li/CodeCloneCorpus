        def get_expected_failures_for?(ids)
          ids_to_run = all_example_ids & (ids + failed_example_ids)
          notify(
            :bisect_individual_run_start,
            :command => shell_command.repro_command_from(ids_to_run),
            :ids_to_run => ids_to_run
          )

          results, duration = track_duration { runner.run(ids_to_run) }
          notify(:bisect_individual_run_complete, :duration => duration, :results => results)

          abort_if_ordering_inconsistent(results)
          (failed_example_ids & results.failed_example_ids) == failed_example_ids
        end

        def setup_request(controller_class_name, action, parameters, session, flash, xhr)
          generated_extras = @routes.generate_extras(parameters.merge(controller: controller_class_name, action: action))
          generated_path = generated_path(generated_extras)
          query_string_keys = query_parameter_names(generated_extras)

          @request.assign_parameters(@routes, controller_class_name, action, parameters, generated_path, query_string_keys)

          @request.session.update(session) if session
          @request.flash.update(flash || {})

          if xhr
            @request.set_header "HTTP_X_REQUESTED_WITH", "XMLHttpRequest"
            @request.fetch_header("HTTP_ACCEPT") do |k|
              @request.set_header k, [Mime[:js], Mime[:html], Mime[:xml], "text/xml", "*/*"].join(", ")
            end

      def format(object)
        if max_formatted_output_length.nil?
          prepare_for_inspection(object).inspect
        else
          formatted_object = prepare_for_inspection(object).inspect
          if formatted_object.length < max_formatted_output_length
            formatted_object
          else
            beginning = truncate_string formatted_object, 0, max_formatted_output_length / 2
            ending = truncate_string formatted_object, -max_formatted_output_length / 2, -1
            beginning + ELLIPSIS + ending
          end

