import example.akka.remote.client.ClientActor;
import example.akka.remote.shared.Messages;
import org.junit.Test;
import org.mockito.Mockito;

import static org.junit.Assert.assertEquals;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.*;

public class Testt {
    @Test
    public void testowanko() throws Exception {

        ClientActor clientActor = mock(ClientActor.class);

        //when( clientActor.findProperModuleStrategy( any() ) ).then( null );

        verifyZeroInteractions( clientActor );

        assertEquals(2+2,4);
    }


}
