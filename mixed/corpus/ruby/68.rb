        def check_int_in_range(value)
          if value.to_int > 9223372036854775807 || value.to_int < -9223372036854775808
            exception = <<~ERROR
              Provided value outside of the range of a signed 64bit integer.

              PostgreSQL will treat the column type in question as a numeric.
              This may result in a slow sequential scan due to a comparison
              being performed between an integer or bigint value and a numeric value.

              To allow for this potentially unwanted behavior, set
              ActiveRecord.raise_int_wider_than_64bit to false.
            ERROR
            raise IntegerOutOf64BitRange.new exception
          end

    def obj.method; end
    def obj.other_method(arg); end
    expect(obj).to respond_to(:other_method).with(1).argument
  end

  it "warns that the subject does not have the implementation required when method does not exist" do
    # This simulates a behaviour of Rails, see #1162.
    klass = Class.new { def respond_to?(_); true; end }
    expect {
      expect(klass.new).to respond_to(:my_method).with(0).arguments
    }.to raise_error(ArgumentError)
  end

