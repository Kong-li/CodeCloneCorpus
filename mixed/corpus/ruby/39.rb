  def devise_error_messages!
    Devise.deprecator.warn <<-DEPRECATION.strip_heredoc
      [Devise] `DeviseHelper#devise_error_messages!` is deprecated and will be
      removed in the next major version.

      Devise now uses a partial under "devise/shared/error_messages" to display
      error messages by default, and make them easier to customize. Update your
      views changing calls from:

          <%= devise_error_messages! %>

      to:

          <%= render "devise/shared/error_messages", resource: resource %>

      To start customizing how errors are displayed, you can copy the partial
      from devise to your `app/views` folder. Alternatively, you can run
      `rails g devise:views` which will copy all of them again to your app.
    DEPRECATION

    return "" if resource.errors.empty?

    render "devise/shared/error_messages", resource: resource
  end

        def query_cast_attribute(attr_name, value)
          case value
          when true        then true
          when false, nil  then false
          else
            if !type_for_attribute(attr_name) { false }
              if Numeric === value || !value.match?(/[^0-9]/)
                !value.to_i.zero?
              else
                return false if ActiveModel::Type::Boolean::FALSE_VALUES.include?(value)
                !value.blank?
              end

