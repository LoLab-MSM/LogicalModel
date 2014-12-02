# Author: Seda Arat
# Name: Basin of Attractors Synch Update
# Revision Date: September 2014

#!/usr/bin/perl

use strict;
use warnings;

##################################################
# To run the example in the handout:
# perl basin_attr_synch.pl -f func-example.txt -n 3

# To run the core iron model with 6 variables and 3 states:
# perl basin_attr_synch.pl -f core_iron_6variables_3states.txt -n 3
##################################################

use Getopt::Euclid;

=head1 NAME

basin_attr_synch.pl - Find the basin of attractors of a given discrete system.

=head1 USAGE

basin_attr_synch.pl -f <functions-file> -n <number-states>

=head1 SYNOPSIS

basin_attr_synch.pl -f <functions-file> -n <number-states>

=head1 DESCRIPTION

basin_attr_synch.pl - Find the basin of attractors of a given discrete system with a synchronous update schedule.

=head1 REQUIRED ARGUMENTS

=over

=item -f[unctions-file] <functions-file>

The name of the file containing the functions or transition table for the finite dynamical system (.txt). 

=for Euclid:

network-file.type: readable

=item -n[umber-of-states] <number-states>

The number of states.

=back

=head1 AUTHOR

Seda Arat

=cut

my $func_file = $ARGV{'-f'};
my $num_states = $ARGV{'-n'};

my @functions = get_functions($func_file);
my $num_nodes = scalar @functions;
my $stateSpace_size = $num_states ** $num_nodes;
my $length = 5;
my %state_attractor_table;
my %attractor_table;

print "stateSpace_size = $stateSpace_size \n";

for (my $is = 1; $is <= $stateSpace_size; $is++) {

  if (exists $state_attractor_table{$is}) {
    next;
  }

  my @array = ($is);
  
  attr: while (1) {
    for (my $n = 1; $n <= $length; $n++) {
      push (@array, get_nextstate ($array[-1]));
    }

    my $arraysize = scalar @array;
    
    for (my $j = 0; $j < $arraysize - 1; $j++) {
      for (my $k = $j + 1; $k < $arraysize; $k++) {
	if ($array[$j] == $array[$k]) {

	  my @sub_array = @array[$j ... $k - 1];
	  my $sortedAttractor = join (' , ', sort {$a <=> $b} @sub_array);

	  unless (exists $attractor_table{$sortedAttractor} ) {
	    my $attractor = join (' -> ', @sub_array);
	    $attractor_table{$sortedAttractor} = [$attractor, 0];
	  }

	  for (my $s = 0; $s < $k; $s++) {
	    my $a = $array[$s];
	    unless (exists $state_attractor_table{$a}) {
	      $state_attractor_table{$a} = $sortedAttractor;
	      ${$attractor_table{$sortedAttractor}}[1]++;
	    }
	  }
	  last attr;
	}
      }
    }

    push (@array, get_nextstate ($array[-1]));

  } # end of while loop
  
} # end of for loop

foreach my $value (values %attractor_table) {
  my @array1 = split (' -> ', $value->[0]);
  my @array2 = ();
  for (my $i = 0; $i < scalar @array1; $i++) {
    $array2[$i] = join (' ', convert_from_integer_to_state($array1[$i]));
  }
  my $attractor = join (' -> ', @array2);
  print "attractor = $attractor and its size = $value->[1] \n";
}

exit;

################################################################################
# Subroutines
################################################################################

#Reads and stores the functions in an array by replacing the external parameters with their values. The first entry is the updating function of the first node, the second entry is the updating function of the second node, so on...

sub get_functions {
  my $file = shift;
  my @functions;
  
  open (FILE, "< $file") or die ("ERROR: Cannot open the functions file for reading! \n");
  while (my $line = <FILE>) {
    chomp ($line);
    
    # skip empty lines
    if ($line =~ /^\s*$/) {
      next;
    }

    if ($line =~ /(f|=|x)/) {
      
      $line =~ s/\^/\*\*/g; # replace carret with double stars
      $line =~ s/x(\d+)/\$x\[$1\]/g; #for evaluation

      my $f;
      
      unless ($line =~ /^f/) {
	my $temp = pop (@functions);
	$f = $temp . $line;
      }
      else {
	(my $a, $f) = split(/=/,$line);
      }

      push(@functions, $f);
    }
    else {
      print ("ERROR: Please revise the format of the functions file. \n");
      exit;
    }
  }
  
  close (FILE) or die ("ERROR: Cannot close the function file! \n");
  
  # Error checking
  
  unless (scalar @functions) {
    print ("ERROR: Please revise the file for functions, which seems empty. \n");
    exit;
  }
  return @functions;
}

################################################################################

# Returns the next state (as an integer) of a given state (as an integer)

sub get_nextstate {
  my $state = shift;

  my @x = convert_from_integer_to_state ($state);
  my @array = ();

  my @temp = @functions;
  
  for (my $i = 0; $i < @temp; $i++) {
    for (my $j = 0; $j < @x; $j++) {
      my $k = $j + 1;
      $temp[$i] =~ s/\$x\[$k\]/\($x[$j]\)/g;
    }
    
    $array[$i] = eval($temp[$i]) % $num_states;
  }

  my $nextState = convert_from_state_to_integer (\@array);
  return $nextState;
}

################################################################################

# Converts a given state (as a ref of array) to its integer representation and adds 1 for convenience.

sub convert_from_state_to_integer {
  my $state = shift;
  my $int_rep = 1;

  for (my $i = 0; $i < $num_nodes; $i++) {
    $int_rep += $state->[$num_nodes - $i - 1] * ($num_states ** $i);
  }
  return $int_rep;
}

################################################################################

# Converts the integer representation of a state to state itself (as an array).

sub convert_from_integer_to_state {
  my $n = shift;

  my ($quotient, $remainder);
  my @state = ();
  $n--;

  do {
    $quotient = int $n / $num_states;
    $remainder = $n % $num_states;
    push (@state, $remainder);
    $n = $quotient;
  } until ($quotient == 0);

  my $dif = $num_nodes - (scalar @state);

  if ($dif) {
    for (my $i = 0; $i < $dif; $i++) {
      push (@state, 0);
    }
  }

  @state = reverse @state;
  return @state;
}

################################################################################
